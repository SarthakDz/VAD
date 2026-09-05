"""Local scorer mirroring the arena's published scheme.

Source: "AHC Visual Intelligence Hackathon Submission format" (2026-09-05).

    Level 1   pooled over all Level-1 videos:
                  0.5 * anomaly-vs-normal accuracy  +  0.5 * class accuracy
    Level 2/3 scored per video, then averaged:
                  gt normal, predicted nothing  -> 1
                  gt normal, predicted anything -> 0
                  gt has events -> weighted mix of alert / matched / timing,
                                   timing weighted higher at Level 3
    An event matches only when the class is right AND IoU >= 0.5.
    One predicted event can match at most one ground-truth event; the rest
    count against you.

ASSUMPTION -- the exact alert/matched/timing weights are not published. The
defaults below are a guess consistent with "timing weighs more at Level 3".
Ask the organisers and set --w2 / --w3. Components are always printed
separately so a change in weights never hides a weak component.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .io_dataset import TEST_COLS, _read_gt
from .labels import NORMAL

IOU_GATE = 0.5

# (alert, matched, timing)
DEFAULT_W2 = (0.2, 0.5, 0.3)
DEFAULT_W3 = (0.2, 0.4, 0.4)


# --------------------------------------------------------------------------
# loading


def load_predictions(path: str | Path) -> dict[str, list[dict]]:
    """video_id -> list of event dicts. Missing video means an empty answer."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for p in doc.get("predictions", []):
        out[str(p["video_id"])] = list(p.get("events") or [])
    return out


def gt_by_video(gt: pd.DataFrame) -> dict[str, dict]:
    """video_id -> {level, is_anomaly, classes, segments}."""
    out: dict[str, dict] = {}
    for vid, sub in gt.groupby("video_id"):
        anom = sub[sub["class_name"] != NORMAL]
        segs = [
            (r.class_name, float(r.start_time_sec), float(r.end_time_sec))
            for r in anom.itertuples()
            if pd.notna(r.start_time_sec) and pd.notna(r.end_time_sec)
        ]
        out[str(vid)] = {
            "level": int(sub["level"].iloc[0]),
            "is_anomaly": bool(sub["is_anomaly"].any()),
            "classes": list(anom["class_name"]),
            "segments": segs,
        }
    return out


# --------------------------------------------------------------------------
# helpers


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _match(gt_segs: list, pred_segs: list) -> list[float]:
    """Greedy one-to-one matching by IoU. Returns the IoU of each match.

    Class must agree and IoU must clear the gate. Extra fragments cannot
    double-match a single ground-truth event -- exactly the behaviour trap 4
    warns about.
    """
    pairs = []
    for pi, (pc, ps, pe) in enumerate(pred_segs):
        for gi, (gc, gs, ge) in enumerate(gt_segs):
            if pc != gc:
                continue
            v = _iou((ps, pe), (gs, ge))
            if v >= IOU_GATE:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, ious = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        ious.append(v)
    return ious


def _pred_segments(events: list[dict]) -> list:
    out = []
    for e in events:
        s, t = e.get("start_time_sec"), e.get("end_time_sec")
        if s is None or t is None:
            continue
        out.append((e.get("class_name"), float(s), float(t)))
    return out


# --------------------------------------------------------------------------
# scoring


def score_level1(gtv: dict, preds: dict) -> dict:
    vids = [v for v, g in gtv.items() if g["level"] == 1]
    if not vids:
        return {"level1": float("nan"), "l1_binary_acc": float("nan"),
                "l1_class_acc": float("nan"), "l1_n": 0}

    bin_hits = cls_hits = 0
    for v in vids:
        g = gtv[v]
        ev = preds.get(v, [])
        pred_anom = len(ev) > 0
        if pred_anom == g["is_anomaly"]:
            bin_hits += 1

        # "One label for the whole clip" -- the first event is the answer;
        # repeating a class earns nothing extra.
        gt_cls = g["classes"][0] if g["classes"] else NORMAL
        pred_cls = ev[0].get("class_name") if ev else NORMAL
        if pred_cls == gt_cls:
            cls_hits += 1

    n = len(vids)
    b, c = bin_hits / n, cls_hits / n
    return {"level1": 0.5 * b + 0.5 * c, "l1_binary_acc": b, "l1_class_acc": c, "l1_n": n}


def score_video_temporal(g: dict, events: list[dict], weights: tuple) -> tuple[float, dict]:
    pred = _pred_segments(events)

    if not g["is_anomaly"]:
        # Trap 5: any prediction on a normal video scores that video zero.
        return (1.0 if not events else 0.0), {"alert": 0.0, "matched": 0.0, "timing": 0.0}

    if not pred:
        return 0.0, {"alert": 0.0, "matched": 0.0, "timing": 0.0}

    ious = _match(g["segments"], pred)
    n_gt, n_pred, n_ok = len(g["segments"]), len(pred), len(ious)

    alert = 1.0
    recall = n_ok / n_gt if n_gt else 0.0
    precision = n_ok / n_pred if n_pred else 0.0
    matched = 2 * precision * recall / (precision + recall) if n_ok else 0.0
    timing = float(np.mean(ious)) if ious else 0.0

    wa, wm, wt = weights
    return wa * alert + wm * matched + wt * timing, {
        "alert": alert, "matched": matched, "timing": timing,
    }


def score_level_n(gtv: dict, preds: dict, level: int, weights: tuple) -> dict:
    vids = [v for v, g in gtv.items() if g["level"] == level]
    if not vids:
        return {f"level{level}": float("nan"), f"l{level}_n": 0}

    totals, comps = [], defaultdict(list)
    for v in vids:
        s, c = score_video_temporal(gtv[v], preds.get(v, []), weights)
        totals.append(s)
        if gtv[v]["is_anomaly"]:
            for k, val in c.items():
                comps[k].append(val)

    out = {f"level{level}": float(np.mean(totals)), f"l{level}_n": len(vids)}
    for k in ("alert", "matched", "timing"):
        out[f"l{level}_{k}"] = float(np.mean(comps[k])) if comps[k] else float("nan")
    return out


def score(gt: pd.DataFrame, preds: dict, w2=DEFAULT_W2, w3=DEFAULT_W3) -> dict:
    gtv = gt_by_video(gt)
    out = {}
    out.update(score_level1(gtv, preds))
    out.update(score_level_n(gtv, preds, 2, w2))
    out.update(score_level_n(gtv, preds, 3, w3))
    vals = [out[k] for k in ("level1", "level2", "level3") if not np.isnan(out.get(k, np.nan))]
    out["overall_mean"] = float(np.mean(vals)) if vals else 0.0
    return out


def confusion_l1(gtv: dict, preds: dict) -> list[tuple[str, str, str]]:
    rows = []
    for v, g in sorted(gtv.items()):
        if g["level"] != 1:
            continue
        ev = preds.get(v, [])
        rows.append((v, g["classes"][0] if g["classes"] else NORMAL,
                     ev[0].get("class_name") if ev else NORMAL))
    return rows


# --------------------------------------------------------------------------


def _weights(s: str | None, default: tuple) -> tuple:
    if not s:
        return default
    parts = tuple(float(x) for x in s.split(","))
    if len(parts) != 3:
        raise SystemExit("weights must be alert,matched,timing")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="submission JSON")
    ap.add_argument("--gt", required=True, help="public test ground_truth.csv")
    ap.add_argument("--w2", help="Level-2 weights 'alert,matched,timing'")
    ap.add_argument("--w3", help="Level-3 weights 'alert,matched,timing'")
    ap.add_argument("--per-video", action="store_true")
    a = ap.parse_args()

    gt = _read_gt(Path(a.gt), TEST_COLS)
    preds = load_predictions(a.pred)
    gtv = gt_by_video(gt)

    answered = sum(1 for v in gtv if v in preds)
    print(f"gt:   {len(gt)} rows / {len(gtv)} videos")
    print(f"pred: {answered} videos answered "
          f"({sum(1 for v in gtv if preds.get(v))} with events)\n")

    res = score(gt, preds, _weights(a.w2, DEFAULT_W2), _weights(a.w3, DEFAULT_W3))

    print(f"  LEVEL 1  {res['level1']:.4f}   (n={res['l1_n']})")
    print(f"      binary_acc {res['l1_binary_acc']:.4f}   class_acc {res['l1_class_acc']:.4f}")
    for lv in (2, 3):
        print(f"  LEVEL {lv}  {res[f'level{lv}']:.4f}   (n={res[f'l{lv}_n']})")
        print(f"      alert {res[f'l{lv}_alert']:.4f}   "
              f"matched {res[f'l{lv}_matched']:.4f}   timing {res[f'l{lv}_timing']:.4f}")
    print(f"\n  OVERALL (unweighted mean) {res['overall_mean']:.4f}")

    if a.per_video:
        print("\n  Level-1 predictions")
        for v, g, p in confusion_l1(gtv, preds):
            print(f"    {v}  gt={g:34s} pred={p}{'' if g == p else '   MISS'}")


if __name__ == "__main__":
    main()
