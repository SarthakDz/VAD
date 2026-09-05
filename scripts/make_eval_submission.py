"""Final submission for the private evaluation set.

Best measured recipe on the public set (49.1/100):
  D1     ensemble of the base and so400m clip classifiers, threshold 0.4.
         The two encoders make different mistakes; averaging them recovered
         11/20 where either alone found 9/20.
  D2/D3  temporal head + hysteresis/merge, enter .92 exit .30 gap 20s min 3s.
         Tuned for precision: the arena penalises false alarms harder than misses.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.clip_classifier import pool
from src.infer_clip import load_clip, Normalised
from src.infer_head import load_head, run as head_run
from src.labels import CLASSES, NORMAL
from src.submit import load_manifest, build, write

MF = load_manifest("data/manifest_eval.json")
DUR = {v["video_id"]: v["duration_sec"]
       for v in json.loads(Path("data/manifest_eval.json").read_text())["videos"]}
dev = "cuda" if torch.cuda.is_available() else "cpu"

t0 = time.perf_counter()
head = load_head("outputs/head.pt", dev)
preds, stats = head_run(MF, Path("cache_eval"), head, dev,
                        enter=0.92, exit_=0.30, merge_gap_sec=20.0, min_event_sec=3.0)

bB, muB, sdB = load_clip("outputs/clip.pt", dev);    cB = Normalised(bB, muB, sdB, dev)
bS, muS, sdS = load_clip("outputs/clip_so.pt", dev); cS = Normalised(bS, muS, sdS, dev)
THR, n_anom = 0.4, 0
for p in preds:
    if MF[p.video_id] != 1:
        continue
    with torch.inference_mode():
        eb = pool(np.load(f"cache_eval/emb/{p.video_id}.npy").astype(np.float32))
        es = pool(np.load(f"cache_eval_so/emb/{p.video_id}.npy").astype(np.float32))
        pb = torch.softmax(cB(torch.from_numpy(eb[None, :]).float().to(dev)), -1)[0].cpu().numpy()
        ps = torch.softmax(cS(torch.from_numpy(es[None, :]).float().to(dev)), -1)[0].cpu().numpy()
    pr = (pb + ps) / 2
    i = int(pr.argmax())
    if CLASSES[i] == NORMAL or pr[i] < THR:
        p.events = []
    else:
        from src.submit import Event
        p.events = [Event(CLASSES[i], None, None)]
        n_anom += 1

# never claim past the end of a video -- a documented rejection
clipped = 0
for p in preds:
    for e in p.events:
        if e.end_time_sec is not None and e.end_time_sec > DUR[p.video_id]:
            e.end_time_sec = round(DUR[p.video_id], 2); clipped += 1
        if e.start_time_sec is not None and e.end_time_sec is not None \
           and e.end_time_sec <= e.start_time_sec:
            e.end_time_sec = round(min(DUR[p.video_id], e.start_time_sec + 1.0), 2)

doc = build(preds, "ahc-final", "siglip-ensemble+gru-cascade",
            (time.perf_counter() - t0) * 1000.0, "1x RTX 4060 Laptop 8GB")
out = write(doc, "outputs/submission_eval.json", MF)

per = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
for p in preds:
    L = MF[p.video_id]; per[L][0] += 1; per[L][1] += len(p.events)
print(f"wrote {out}")
for L in (1, 2, 3):
    print(f"  D{L}: {per[L][0]} videos, {per[L][1]} events")
print(f"  D1 anomalies claimed: {n_anom}/20   timestamps clipped to duration: {clipped}")
print(f"  {stats['video_sec']/60:.1f} min of video, {stats['realtime_factor']:.1f}x realtime")
