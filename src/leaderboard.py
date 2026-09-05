"""Scoring in the arena leaderboard's own units.

Read off the live leaderboard on 2026-09-05 at 12:27, which supersedes the
equal-weighted guess in `score.py`:

    D1  "Clear event"     out of 25    which anomalies appear
    D2  "When it happens" out of 35    class, anomaly and timing
    D3  "Long context"    out of 40    all of it, over long footage
    REASON bonus sits in its own column so it never hides weak accuracy.
    "A difficulty that is never submitted scores zero."

**Level 3 is worth 40% of the total and is our weakest level.** Any tuning that
trades Level 3 away for Level 1 is going the wrong direction.

The leaderboard reports, per difficulty: marks, P, R, found (x/y) and FA. The
exact marks formula is not published, but two rows of evidence pin down its
shape, and the conclusion is strong enough to tune on:

    Yash Waghmare  D2  P 100%  R 22%  found 4/18  FA 0  ->  29.9/35  (85%)
    Aryan Varale   D2  P  38%  R 28%  found 5/18  FA 8  ->  25.1/35  (72%)

**Aryan found MORE events than Yash and scored LOWER.** Eight false alarms cost
more than one extra detection gained. And Yash reached 85% of Difficulty 2 while
missing 78% of the events, purely on perfect precision.

So the objective is precision and false-alarm suppression, not recall. Emit few,
confident events. This module therefore reports P / R / found / FA directly,
in the leaderboard's units, so our numbers can be read against theirs. `marks`
is an F1-based proxy and is explicitly NOT the official formula -- it fits D1
(Yash 25.0, Aryan ~10.3 against an actual 10.6) but underestimates D2 and D3,
where the real scheme is evidently kinder to low recall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import NORMAL
from .score import _iou, gt_by_video

MAX_POINTS = {1: 25.0, 2: 35.0, 3: 40.0}
IOU_GATE = 0.5


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def difficulty1(gtv: dict, preds: dict) -> dict:
    """Video-level. A video counts as found when it is anomalous and the
    predicted class matches; anything else predicted anomalous is a false
    alarm, whether it is a wrong class or a normal video called anomalous."""
    found = fa = n_gt = 0
    for vid, g in gtv.items():
        if g["level"] != 1:
            continue
        ev = preds.get(vid, [])
        pred_cls = ev[0].get("class_name") if ev else NORMAL
        gt_cls = g["classes"][0] if g["classes"] else NORMAL
        if gt_cls != NORMAL:
            n_gt += 1
        if pred_cls == NORMAL:
            continue
        if pred_cls == gt_cls:
            found += 1
        else:
            fa += 1
    return _pack(1, found, fa, n_gt)


def difficulty_n(gtv: dict, preds: dict, level: int) -> dict:
    """Event-level, with the class correct and IoU >= 0.5. One predicted event
    may match at most one ground-truth event; every unmatched prediction is a
    false alarm, including anything predicted on a normal video."""
    found = fa = n_gt = 0
    for vid, g in gtv.items():
        if g["level"] != level:
            continue
        n_gt += len(g["segments"])
        pred = [(e.get("class_name"), e.get("start_time_sec"), e.get("end_time_sec"))
                for e in preds.get(vid, [])]
        pred = [(c, float(s), float(e)) for c, s, e in pred if s is not None and e is not None]

        pairs = []
        for pi, (pc, ps, pe) in enumerate(pred):
            for gi, (gc, gs, ge) in enumerate(g["segments"]):
                if pc == gc:
                    v = _iou((ps, pe), (gs, ge))
                    if v >= IOU_GATE:
                        pairs.append((v, pi, gi))
        pairs.sort(reverse=True)
        used_p, used_g = set(), set()
        for _, pi, gi in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
        found += len(used_g)
        fa += len(pred) - len(used_p)
    return _pack(level, found, fa, n_gt)


def _pack(level: int, found: int, fa: int, n_gt: int) -> dict:
    p = found / (found + fa) if (found + fa) else 0.0
    r = found / n_gt if n_gt else 0.0
    return {
        "level": level, "found": found, "n_gt": n_gt, "fa": fa,
        "precision": p, "recall": r,
        "marks_proxy": MAX_POINTS[level] * _f1(p, r),
        "max_points": MAX_POINTS[level],
    }


def report(gt: pd.DataFrame, preds: dict) -> dict:
    gtv = gt_by_video(gt)
    rows = [difficulty1(gtv, preds), difficulty_n(gtv, preds, 2),
            difficulty_n(gtv, preds, 3)]
    total = sum(r["marks_proxy"] for r in rows)
    return {"difficulties": rows, "total_proxy": total}


def print_report(gt: pd.DataFrame, preds: dict, label: str = "") -> dict:
    r = report(gt, preds)
    names = {1: "D1 Clear event", 2: "D2 When it happens", 3: "D3 Long context"}
    if label:
        print(f"  {label}")
    print(f"  {'':22s}{'P':>7}{'R':>7}{'found':>9}{'FA':>5}{'marks~':>9}{'max':>6}")
    for d in r["difficulties"]:
        print(f"  {names[d['level']]:22s}{d['precision']*100:6.0f}%{d['recall']*100:6.0f}%"
              f"{d['found']:6d}/{d['n_gt']:<2d}{d['fa']:5d}"
              f"{d['marks_proxy']:9.1f}{d['max_points']:6.0f}")
    print(f"  {'TOTAL (proxy, /100)':22s}{'':7}{'':7}{'':9}{'':5}{r['total_proxy']:9.1f}{100:6.0f}")
    return r


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from .io_dataset import TEST_COLS, _read_gt
    from .score import load_predictions

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default="../Train and Test/test/ground_truth.csv")
    a = ap.parse_args()
    print_report(_read_gt(Path(a.gt), TEST_COLS), load_predictions(a.pred))
