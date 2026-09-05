"""Local scorer with a pluggable metric interface.

The private metric is not published -- it is revealed on the day. So this
implements several plausible ones behind `--metric` and prints them all by
default. Swap the primary once the organisers answer.

Metrics
  binary_f1     video-level anomaly / not-anomaly F1
  video_acc     primary-class accuracy (one class per video)
  macro_f1      multilabel macro-F1 over the 12 classes, per video
  per_class_f1  the same, broken out per class
  tiou_f1       segment-level F1 at IoU thresholds, averaged (Levels 2-3)
  frame_f1      per-second class labelling F1 -- the most forgiving temporal view
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .io_dataset import SUBMISSION_COLS, TEST_COLS, _read_gt
from .labels import ANOMALY_CLASSES, NORMAL

TIOU_THRESHOLDS = (0.1, 0.3, 0.5)
FRAME_STEP_SEC = 1.0

METRICS = {}


def metric(name):
    def deco(fn):
        METRICS[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------
# helpers

def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def _video_classes(df: pd.DataFrame) -> dict:
    """video_id -> set of anomaly classes claimed for it."""
    out = defaultdict(set)
    for vid, cls in zip(df["video_id"], df["class_name"]):
        if cls != NORMAL:
            out[vid].add(cls)
    return out


def _primary_class(df: pd.DataFrame) -> dict:
    """video_id -> the single class that best represents it.

    Longest total duration wins; falls back to first-listed for Level 1 rows
    that carry no timestamps.
    """
    best = {}
    for row in df.itertuples():
        if row.class_name == NORMAL:
            best.setdefault(row.video_id, (-1.0, NORMAL))
            continue
        dur = 0.0
        if pd.notna(row.start_time_sec) and pd.notna(row.end_time_sec):
            dur = float(row.end_time_sec) - float(row.start_time_sec)
        cur = best.get(row.video_id)
        if cur is None or dur > cur[0]:
            best[row.video_id] = (dur, row.class_name)
    return {v: c for v, (_, c) in best.items()}


def _segments(df: pd.DataFrame) -> dict:
    """(video_id, class_name) -> list of (start, end), anomalies only."""
    out = defaultdict(list)
    for row in df.itertuples():
        if row.class_name == NORMAL:
            continue
        if pd.isna(row.start_time_sec) or pd.isna(row.end_time_sec):
            continue
        out[(row.video_id, row.class_name)].append(
            (float(row.start_time_sec), float(row.end_time_sec))
        )
    return out


def _iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# metrics

@metric("binary_f1")
def binary_f1(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    g = gt.groupby("video_id")["is_anomaly"].any()
    p = pred.groupby("video_id")["is_anomaly"].any().reindex(g.index, fill_value=False)
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())
    tn = int((~g & ~p).sum())
    return {
        "binary_f1": _f1(tp, fp, fn),
        "binary_acc": (tp + tn) / max(1, len(g)),
        "b_tp": tp, "b_fp": fp, "b_fn": fn, "b_tn": tn,
    }


@metric("video_acc")
def video_acc(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    g, p = _primary_class(gt), _primary_class(pred)
    hits = sum(1 for v, c in g.items() if p.get(v, NORMAL) == c)
    return {"video_acc": hits / max(1, len(g)), "n_videos": len(g)}


@metric("per_class_f1")
def per_class_f1(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    g, p = _video_classes(gt), _video_classes(pred)
    vids = set(gt["video_id"])
    per = {}
    for c in ANOMALY_CLASSES:
        tp = sum(1 for v in vids if c in g.get(v, set()) and c in p.get(v, set()))
        fp = sum(1 for v in vids if c not in g.get(v, set()) and c in p.get(v, set()))
        fn = sum(1 for v in vids if c in g.get(v, set()) and c not in p.get(v, set()))
        per[c] = {"f1": _f1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}
    return {"_per_class": per}


@metric("macro_f1")
def macro_f1(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    per = per_class_f1(gt, pred)["_per_class"]
    present = [c for c in ANOMALY_CLASSES if per[c]["support"] > 0]
    return {
        "macro_f1": float(np.mean([per[c]["f1"] for c in present])) if present else 0.0,
        "n_classes_present": len(present),
    }


@metric("tiou_f1")
def tiou_f1(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    gs, ps = _segments(gt), _segments(pred)
    out = {}
    for thr in TIOU_THRESHOLDS:
        tp = fp = fn = 0
        for key in set(gs) | set(ps):
            gsegs, psegs = list(gs.get(key, [])), list(ps.get(key, []))
            used = set()
            for pseg in psegs:
                best_i, best_v = -1, thr
                for i, gseg in enumerate(gsegs):
                    if i in used:
                        continue
                    v = _iou(pseg, gseg)
                    if v >= best_v:
                        best_i, best_v = i, v
                if best_i >= 0:
                    used.add(best_i)
                    tp += 1
                else:
                    fp += 1
            fn += len(gsegs) - len(used)
        out["tiou_f1@" + str(thr)] = _f1(tp, fp, fn)
    out["tiou_f1_avg"] = float(np.mean(list(out.values())))
    return out


@metric("frame_f1")
def frame_f1(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Quantise both sides to 1-second bins and compare class labels."""
    horizon = defaultdict(float)
    for df in (gt, pred):
        for row in df.itertuples():
            if pd.notna(row.end_time_sec):
                horizon[row.video_id] = max(horizon[row.video_id], float(row.end_time_sec))

    tp = fp = fn = 0
    for vid, end in horizon.items():
        n = max(1, int(np.ceil(end / FRAME_STEP_SEC)))
        gl = [set() for _ in range(n)]
        pl = [set() for _ in range(n)]
        for df, lanes in ((gt, gl), (pred, pl)):
            sub = df[df["video_id"] == vid]
            for row in sub.itertuples():
                if row.class_name == NORMAL or pd.isna(row.start_time_sec):
                    continue
                lo = int(np.floor(float(row.start_time_sec) / FRAME_STEP_SEC))
                hi = int(np.ceil(float(row.end_time_sec) / FRAME_STEP_SEC))
                for t in range(max(0, lo), min(n, hi)):
                    lanes[t].add(row.class_name)
        for a, b in zip(gl, pl):
            tp += len(a & b)
            fp += len(b - a)
            fn += len(a - b)
    return {"frame_f1": _f1(tp, fp, fn), "f_tp": tp, "f_fp": fp, "f_fn": fn}


# --------------------------------------------------------------------------

def score_all(gt: pd.DataFrame, pred: pd.DataFrame, which=None) -> dict:
    names = which or [n for n in METRICS if n != "per_class_f1"]
    out = {}
    for n in names:
        out.update({k: v for k, v in METRICS[n](gt, pred).items() if not k.startswith("_")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", action="append", choices=sorted(METRICS),
                    help="repeatable; default is every metric")
    ap.add_argument("--per-class", action="store_true")
    a = ap.parse_args()

    gt = _read_gt(Path(a.gt), TEST_COLS)
    pred = _read_gt(Path(a.pred), SUBMISSION_COLS)

    print("gt:   %4d rows / %d videos" % (len(gt), gt["video_id"].nunique()))
    print("pred: %4d rows / %d videos\n" % (len(pred), pred["video_id"].nunique()))

    for k, v in score_all(gt, pred, a.metric).items():
        if isinstance(v, float):
            print("  %-20s %.4f" % (k, v))
        else:
            print("  %-20s %s" % (k, v))

    if a.per_class:
        print("\n  class                              f1     tp   fp   fn  supp")
        for c, m in per_class_f1(gt, pred)["_per_class"].items():
            print("  %-32s %.3f  %4d %4d %4d  %4d"
                  % (c, m["f1"], m["tp"], m["fp"], m["fn"], m["support"]))


if __name__ == "__main__":
    main()
