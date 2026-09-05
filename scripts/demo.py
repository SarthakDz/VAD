"""Live presentation demo: run the real pipeline on one video and show the answer.

Reads a video's cached embeddings (produced once by src/encode.py, never
touched live) and runs the actual trained head + segment logic used for every
real submission -- nothing here is faked or pre-baked. Takes under a second
per video because the expensive step (decode + encode) already happened.

Usage:
    ./.venv/Scripts/python.exe scripts/demo.py T012          # Level 1 clip
    ./.venv/Scripts/python.exe scripts/demo.py T025          # Level 2 video
    ./.venv/Scripts/python.exe scripts/demo.py T012 T013 T014 T006   # several

If no video_id is given, runs a small default set that tells a good story on
stage: four correct Level-1 detections (fire, fire, smoke, accident) plus one
Level-2 video where all four events are found within a second of the true
boundary, correct class every time.

T025 is deliberately not in the default set: it finds all six events at nearly
the right time but mislabels the class (calls it wrong_way_driving instead of
traffic_accident) -- a real, known limitation, not a demo bug. Run it on
purpose if you want an honest "here is what we have not fixed yet" moment:
    ./.venv/Scripts/python.exe scripts/demo.py T025
"""
import sys, json, io
from pathlib import Path
# Windows consoles default to cp1252, which cannot print block-drawing
# characters -- force UTF-8 so this never crashes mid-demo on stage.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.frames import VideoMeta, timestamps
from src.head import predict as head_predict
from src.infer_head import load_head
from src.labels import CLASSES, NORMAL
from src.segments import extract, to_events

CACHE = "cache"          # public set -- ground truth known, good for a demo
DEFAULT = ["T012", "T013", "T014", "T006", "T028"]

dev = "cuda" if torch.cuda.is_available() else "cpu"
head = load_head("outputs/head.pt", dev)


def bar(v, width=40):
    n = int(round(v * width))
    return "#" * n + "-" * (width - n)


def run(vid):
    emb_path = Path(f"{CACHE}/emb/{vid}.npy")
    if not emb_path.exists():
        print(f"  [skip] {vid}: not cached")
        return
    emb = np.load(emb_path).astype(np.float32)
    m = json.loads(Path(f"{CACHE}/meta/{vid}.json").read_text())
    vm = VideoMeta(vid, m["duration_sec"], m["native_fps"], m["native_frames"], m["width"],
                    m["height"], m["sampled_frames"], m["sample_fps"], int(m.get("frame_step", 1)))

    t0 = __import__("time").perf_counter()
    a, c = head_predict(head, torch.from_numpy(emb), dev)
    a, c, ts = a.numpy(), c.numpy(), timestamps(vm)
    ms = (__import__("time").perf_counter() - t0) * 1000

    print(f"\n{'=' * 78}\n  {vid}   {m['duration_sec']:.1f}s video   "
          f"{len(a)} scored timesteps   head ran in {ms:.1f} ms\n{'=' * 78}")

    # a compact ASCII sparkline of the anomaly score across the whole video --
    # this is the exact curve the segment logic below acts on
    n_buckets = 60
    bucket = np.array_split(a, min(n_buckets, len(a)))
    levels = " .:-=+*#%@"                       # ASCII-only: safe on every console
    spark = "".join(levels[min(9, int(b.mean() * 10))] for b in bucket)
    print(f"  anomaly over time   0s [{spark}] {m['duration_sec']:.0f}s")
    print(f"                          {'low':<{len(spark)-3}}high")

    segs = extract(a, c, ts, enter=0.70, exit_=0.45, merge_gap_sec=5.0, min_event_sec=2.0)
    events = to_events(segs, level=2 if m["duration_sec"] > 30 else 1)

    if not events:
        print("  -> NORMAL (no event above threshold)")
        return
    for e in events:
        if e.start_time_sec is None:
            print(f"  -> {e.class_name}   (whole-clip label, Level 1)")
        else:
            print(f"  -> {e.class_name:32s} {e.start_time_sec:6.1f}s - {e.end_time_sec:6.1f}s   "
                  f"({e.end_time_sec - e.start_time_sec:.1f}s)")


if __name__ == "__main__":
    vids = sys.argv[1:] or DEFAULT
    for v in vids:
        run(v)
    print()
