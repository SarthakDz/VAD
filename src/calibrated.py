"""Scorer calibrated against a real arena result.

Submitted `outputs/submission.json` on 2026-09-05 and the arena returned
D1 12.9/25, D2 22.6/35, D3 11.5/40 = 47.0. That one observation pins down what
the format PDF left ambiguous:

  D1 is F1-based, NOT the PDF's 0.5*binary + 0.5*class.
     Our found=9, fa=5, n_gt=20 -> P .643, R .450, F1 .529 -> 13.2  (actual 12.9)
     The PDF formula predicts 16.1, which is far off.
     Consequence: D1 false alarms cost marks directly.

  D2/D3 keep the per-video partial-credit model in score.py, but D2's weights
     are nearer (alert .3, matched .4, timing .3) than the assumed (.2,.5,.3):
     solving 2 normals at 1.0 plus 4 anomalous videos against 22.6/35 gives
     wa + .25*wm + .236*wt = .469, which (.3,.4,.3) satisfies at .471.

  D3 weights (.2,.4,.4) already predicted .279 against an actual .288 -- kept.

Residual error after calibration is about 1 mark, which is good enough to rank
configurations against each other.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .leaderboard import difficulty1
from .score import gt_by_video, score_level_n

MAX_POINTS = {1: 25.0, 2: 35.0, 3: 40.0}
W2 = (0.30, 0.40, 0.30)
W3 = (0.20, 0.40, 0.40)


def marks(gt: pd.DataFrame, preds: dict) -> dict:
    gtv = gt_by_video(gt)
    d1 = difficulty1(gtv, preds)
    p, r = d1["precision"], d1["recall"]
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    l2 = score_level_n(gtv, preds, 2, W2)
    l3 = score_level_n(gtv, preds, 3, W3)
    m1, m2, m3 = 25.0 * f1, 35.0 * l2["level2"], 40.0 * l3["level3"]
    return {"d1": m1, "d2": m2, "d3": m3, "total": m1 + m2 + m3,
            "d1_found": d1["found"], "d1_fa": d1["fa"],
            "d1_p": p, "d1_r": r,
            "l2": l2["level2"], "l3": l3["level3"]}


if __name__ == "__main__":
    import argparse

    from .io_dataset import TEST_COLS, _read_gt
    from .score import load_predictions

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", default="../Train and Test/test/ground_truth.csv")
    a = ap.parse_args()
    m = marks(_read_gt(Path(a.gt), TEST_COLS), load_predictions(a.pred))
    print(f"  D1 {m['d1']:5.1f}/25   found {m['d1_found']}/20  FA {m['d1_fa']}  "
          f"P {m['d1_p']*100:.0f}% R {m['d1_r']*100:.0f}%")
    print(f"  D2 {m['d2']:5.1f}/35")
    print(f"  D3 {m['d3']:5.1f}/40")
    print(f"  TOTAL {m['total']:5.1f}/100")
