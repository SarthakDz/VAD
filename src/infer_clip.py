"""Submission built from the clip classifier.

D1     pool the whole clip, classify once. This is exactly the task the
       difficulty asks for and exactly what the classifier was trained on.
D2/D3  slide a window of training-clip length, classify each, merge runs of the
       same confident class into events.

Falls back to the temporal head for nothing -- this is a complete alternative
Stage A. Compare against `infer_head` with `src.calibrated`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .clip_classifier import ClipHead, classify_windows, pool, windows_to_segments
from .frames import VideoMeta, timestamps
from .labels import CLASS_TO_IDX, CLASSES, NORMAL
from .segments import to_events
from .submit import (ModelRuntime, RuntimeMetadata, VideoPrediction, build,
                     load_manifest, write)

ENCODER_NAME = "siglip-frozen-encoder"
CLIP_NAME = "clip-classifier"


def load_clip(path: str | Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ClipHead(**ck["config"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck["mu"], ck["sd"]


class Normalised(torch.nn.Module):
    """Wraps the head so callers can feed raw pooled features."""

    def __init__(self, model, mu, sd, device):
        super().__init__()
        self.model = model
        self.mu = torch.from_numpy(mu).float().to(device)
        self.sd = torch.from_numpy(sd).float().to(device)

    def forward(self, x):
        return self.model((x - self.mu) / self.sd)


def run(manifest, cache: Path, model, device: str, win_sec: float, stride_sec: float,
        threshold: float, d1_threshold: float, min_run: int, merge_gap_sec: float,
        min_event_sec: float):
    preds, stats = [], {"videos": 0, "events": 0, "video_sec": 0.0, "internal_ms": 0.0}
    for vid, level in manifest.items():
        ep, mp = cache / "emb" / f"{vid}.npy", cache / "meta" / f"{vid}.json"
        if not ep.exists() or not mp.exists():
            preds.append(VideoPrediction(video_id=vid))
            continue
        emb = np.load(ep).astype(np.float32)
        meta = json.loads(mp.read_text())
        vm = VideoMeta(vid, meta["duration_sec"], meta["native_fps"], meta["native_frames"],
                       meta["width"], meta["height"], meta["sampled_frames"],
                       meta["sample_fps"], int(meta.get("frame_step", 1)))
        ts = timestamps(vm)

        t0 = time.perf_counter()
        if level == 1:
            with torch.inference_mode():
                x = torch.from_numpy(pool(emb)[None, :]).float().to(device)
                p = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
            idx = int(p.argmax())
            # Abstain rather than guess: on D1 a wrong class is a false alarm and
            # costs precision, while an empty answer only costs recall. The
            # difficulty is F1-scored, so a low-confidence guess is negative EV.
            if CLASSES[idx] == NORMAL or p[idx] < d1_threshold:
                events = []
            else:
                from .submit import Event
                events = [Event(CLASSES[idx], None, None)]
        else:
            centres, probs = classify_windows(emb, model, ts, win_sec, stride_sec, device)
            segs = windows_to_segments(centres, probs, ts, threshold, min_run,
                                       merge_gap_sec, min_event_sec)
            events = to_events(segs, level)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        dec = float(meta.get("decode_sec", 0.0)) * 1000.0
        enc = float(meta.get("encode_sec", 0.0)) * 1000.0
        internal = dec + enc + infer_ms
        stats["videos"] += 1
        stats["events"] += len(events)
        stats["video_sec"] += vm.duration_sec
        stats["internal_ms"] += internal
        preds.append(VideoPrediction(vid, events, RuntimeMetadata(
            vm.sampled_frames, 1, internal,
            [ModelRuntime(ENCODER_NAME, vm.sampled_frames, enc),
             ModelRuntime(CLIP_NAME, 1, infer_ms)])))
    return preds, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Train and Test")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--clip", default="outputs/clip.pt")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--out", default="outputs/submission_clip.json")
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--d1-threshold", type=float, default=0.0)
    ap.add_argument("--min-run", type=int, default=2)
    ap.add_argument("--merge-gap", type=float, default=10.0)
    ap.add_argument("--min-event", type=float, default=2.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()

    mf = load_manifest(a.manifest)
    base, mu, sd = load_clip(a.clip, a.device)
    model = Normalised(base, mu, sd, a.device)

    t0 = time.perf_counter()
    preds, st = run(mf, Path(a.cache), model, a.device, a.win, a.stride, a.threshold,
                    a.d1_threshold, a.min_run, a.merge_gap, a.min_event)
    doc = build(preds, "stage-a-clip", "siglip-clip-classifier",
                (time.perf_counter() - t0) * 1000.0, "1x RTX 4060 Laptop 8GB")
    p = write(doc, a.out, mf)
    rt = st["video_sec"] / max(st["internal_ms"] / 1000.0, 1e-9)
    print(f"wrote {p}  {st['videos']} videos, {st['events']} events, {rt:.1f}x realtime")

    if a.score:
        from .calibrated import marks
        from .io_dataset import TEST_COLS, _read_gt
        from .score import load_predictions
        gt = _read_gt(Path(a.root) / "test" / "ground_truth.csv", TEST_COLS)
        m = marks(gt, load_predictions(p))
        print(f"  D1 {m['d1']:5.1f}/25  found {m['d1_found']}/20 FA {m['d1_fa']}  "
              f"P {m['d1_p']*100:.0f}% R {m['d1_r']*100:.0f}%")
        print(f"  D2 {m['d2']:5.1f}/35   D3 {m['d3']:5.1f}/40   TOTAL {m['total']:5.1f}/100")


if __name__ == "__main__":
    main()
