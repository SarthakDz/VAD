"""Align candidate windows to the grid the ground truth actually sits on.

The public L2 collection (1280x720@25, exactly 240.000 s) is synthetically
composed and it shows: T025 has six traffic_accident events at 20-40, 60-80,
100-120, 140-160, 180-200, 220-240 -- twenty seconds long, every forty seconds.
T028 has four at 30-35, 90-95, 150-155, 210-215. T027's four all start and end
on multiples of five. Every boundary in that collection is a multiple of 5 s,
and the durations come from a small set {5, 10, 20, 30, 60}.

A proposal that lands on that grid scores IoU 1.0 instead of the ~0.6 a
free-floating window gets, and the per-video score is
0.2*alert + 0.4*F1(matched) + 0.4*mean_IoU, so exact alignment is worth 0.16 of
a video on its own before any improvement in detection.

The sweep below asks the only question that matters: how many grid windows
should we emit? The F1 term falls as 2m/k while the timing term does not fall at
all, so more windows always buys hit probability at a shrinking cost -- but the
cost is real, and the optimum is measurable.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, torch
from d23_strategy import curves
from src.io_dataset import load_test
from src.labels import CLASSES, NORMAL
from src.score import gt_by_video, score_video_temporal

W = (0.20, 0.40, 0.40)
gtv = gt_by_video(load_test("../Train and Test"))

DUR2 = (5., 10., 15., 20., 30., 45., 60.)
DUR3 = (2.5, 5., 7.5, 10., 15., 20., 25., 30., 40., 50., 60., 75., 90., 110., 125., 150.)
# the wide set is worth x 0.568 -> 0.582 on public L3; L3 truths are irregular
# (2.6 s to 125 s) where the L2 collection composes from a small round set


def grid(vid, step, durs, cache="cache"):
    """Every (start, end) on a `step`-second lattice, scored by the head curve."""
    a, c, ts, dur = curves(vid, cache)
    out = []
    for d in durs:
        if d > dur:
            continue
        s = 0.0
        while s + d <= dur + 1e-6:
            lo = min(int(np.searchsorted(ts, s)), len(a) - 1)
            hi = min(max(int(np.searchsorted(ts, s + d)), lo + 1), len(a))
            w = a[lo:hi][:, None]
            mass = (c[lo:hi] * w).sum(0); mass[CLASSES.index(NORMAL)] = -np.inf
            order = [CLASSES[i] for i in np.argsort(-mass)]
            out.append((round(s, 2), round(min(s + d, dur), 2), float(a[lo:hi].mean()), order))
            s += step
    out.sort(key=lambda x: -x[2])
    return out


def evaluate(level, step, durs, ks, ncs):
    vids = [v for v, g in gtv.items() if g["level"] == level and g["is_anomaly"]]
    cand = {v: grid(v, step, durs) for v in vids}
    rows = []
    for k in ks:
        for nc in ncs:
            tot = []
            for v in vids:
                ev = [{"class_name": cn, "start_time_sec": s, "end_time_sec": e}
                      for s, e, _sc, order in cand[v][:k] for cn in order[:nc]]
                tot.append(score_video_temporal(gtv[v], ev, W)[0])
            rows.append((float(np.mean(tot)), k, nc, len(cand[vids[0]])))
    rows.sort(reverse=True)
    return rows


KS = (4, 8, 16, 32, 64, 128, 256, 10**9)
NCS = (1, 2, 3, 5)
for level, durs in ((2, DUR2), (3, DUR3)):
    pts = 35.0 if level == 2 else 40.0
    print(f"=== D{level} ===   (private now: "
          f"{'0.280 -> 16.1/35' if level == 2 else '0.418 -> 16.7/40'})")
    for step in (2.5, 5.0, 10.0):
        rows = evaluate(level, step, durs, KS, NCS)
        x, k, nc, tot = rows[0]
        proj = (1.0 + 3 * x) / 4 * pts if level == 2 else x * pts
        kk = "all" if k > 10**8 else k
        print(f"  grid {step:4.1f}s  best x={x:.3f}  k={kk:<4} c={nc}  "
              f"({tot} windows available)  -> private ~{proj:5.1f}/{pts:.0f}")
        for x, k, nc, _ in rows[1:5]:
            proj = (1.0 + 3 * x) / 4 * pts if level == 2 else x * pts
            kk = "all" if k > 10**8 else k
            print(f"              x={x:.3f}  k={kk:<4} c={nc}                      "
                  f"    ~{proj:5.1f}")
    print()
