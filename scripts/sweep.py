"""Sweep segment parameters over cached score curves.

`src.infer_head` writes cache/scores/{video_id}.npy, so the head never has to
run again. A full sweep is pure numpy and takes seconds, which is the point:
the fragmented-oracle result says segment shaping decides the Level 2/3 score,
so this is the loop worth running dozens of times.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.frames import VideoMeta, timestamps  # noqa: E402
from src.io_dataset import TEST_COLS, _read_gt  # noqa: E402
from src.score import gt_by_video, score  # noqa: E402
from src.segments import extract, to_events  # noqa: E402
from src.submit import manifest_from_public_test  # noqa: E402


def load_all(cache: Path, manifest: dict):
    out = {}
    for vid in manifest:
        sp = cache / "scores" / f"{vid}.npy"
        mp = cache / "meta" / f"{vid}.json"
        if not sp.exists() or not mp.exists():
            continue
        arr = np.load(sp).astype(np.float32)
        meta = json.loads(mp.read_text())
        vm = VideoMeta(vid, meta["duration_sec"], meta["native_fps"],
                       meta["native_frames"], meta["width"], meta["height"],
                       meta["sampled_frames"], meta["sample_fps"],
                       int(meta.get("frame_step", 1)))
        out[vid] = (arr[:, 0], arr[:, 1:], timestamps(vm))
    return out


def predict(curves, manifest, enter, exit_, gap, min_ev):
    preds = {}
    for vid, (a, c, ts) in curves.items():
        segs = extract(a, c, ts, enter, exit_, gap, min_ev)
        preds[vid] = [e.to_json() for e in to_events(segs, manifest[vid])]
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Train and Test")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--enter", default="0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--exit", dest="exit_", default="0.2,0.3,0.45")
    ap.add_argument("--gap", default="2,5,10,20,40")
    ap.add_argument("--min-event", default="1,2,5")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--sort", default="overall_mean")
    a = ap.parse_args()

    mf = manifest_from_public_test(a.root)
    gt = _read_gt(Path(a.root) / "test" / "ground_truth.csv", TEST_COLS)
    gtv = gt_by_video(gt)
    curves = load_all(Path(a.cache), mf)
    print(f"{len(curves)} cached score curves, {len(gtv)} gt videos\n")

    grid = list(itertools.product(
        [float(x) for x in a.enter.split(",")],
        [float(x) for x in a.exit_.split(",")],
        [float(x) for x in a.gap.split(",")],
        [float(x) for x in a.min_event.split(",")],
    ))

    rows = []
    for en, ex, gp, mn in grid:
        if ex >= en:
            continue
        r = score(gt, predict(curves, mf, en, ex, gp, mn))
        rows.append({"enter": en, "exit": ex, "gap": gp, "min_ev": mn, **r})

    rows.sort(key=lambda r: r[a.sort], reverse=True)
    print(f"{len(rows)} configs, top {a.top} by {a.sort}\n")
    hdr = (f"{'enter':>6}{'exit':>6}{'gap':>6}{'min':>5} | {'L1':>6}{'L2':>7}{'L3':>7}"
           f"{'OVER':>7} | {'l2match':>8}{'l2time':>7}{'l3match':>8}{'l3time':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows[: a.top]:
        print(f"{r['enter']:6.2f}{r['exit']:6.2f}{r['gap']:6.0f}{r['min_ev']:5.0f} | "
              f"{r['level1']:6.3f}{r['level2']:7.3f}{r['level3']:7.3f}{r['overall_mean']:7.3f} | "
              f"{r['l2_matched']:8.3f}{r['l2_timing']:7.3f}"
              f"{r['l3_matched']:8.3f}{r['l3_timing']:7.3f}")

    best = rows[0]
    print(f"\nbest: --enter {best['enter']} --exit {best['exit']} "
          f"--merge-gap {best['gap']} --min-event {best['min_ev']}")
    Path("outputs").mkdir(exist_ok=True)
    json.dump(rows[:50], open("outputs/sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
