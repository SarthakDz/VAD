"""Sanity-check the scorer: a perfect answer must score 1.0, cheats must not."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_dataset import TEST_COLS, _read_gt  # noqa: E402
from src.labels import ANOMALY_CLASSES, NORMAL  # noqa: E402
from src.score import gt_by_video, score  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "../Train and Test")
gt = _read_gt(ROOT / "test" / "ground_truth.csv", TEST_COLS)
gtv = gt_by_video(gt)


def oracle():
    out = {}
    for v, g in gtv.items():
        if not g["is_anomaly"]:
            out[v] = []
        elif g["level"] == 1:
            out[v] = [{"class_name": g["classes"][0],
                       "start_time_sec": None, "end_time_sec": None}]
        else:
            out[v] = [{"class_name": c, "start_time_sec": s, "end_time_sec": e}
                      for c, s, e in g["segments"]]
    return out


def whole_clip():
    """Trap 6: claim the entire clip is one long accident."""
    out = {}
    for v, g in gtv.items():
        if g["level"] == 1:
            out[v] = [{"class_name": "traffic_accident",
                       "start_time_sec": None, "end_time_sec": None}]
        else:
            end = max((e for _, _, e in g["segments"]), default=60.0) * 1.5
            out[v] = [{"class_name": "traffic_accident",
                       "start_time_sec": 0.0, "end_time_sec": end}]
    return out


def spam_classes():
    """Emit every anomaly class on every video."""
    out = {}
    for v, g in gtv.items():
        if g["level"] == 1:
            out[v] = [{"class_name": c, "start_time_sec": None, "end_time_sec": None}
                      for c in ANOMALY_CLASSES]
        else:
            out[v] = [{"class_name": c, "start_time_sec": 0.0, "end_time_sec": 30.0}
                      for c in ANOMALY_CLASSES]
    return out


def fragmented():
    """Trap 4: shatter each true event into 5 slices."""
    out = {}
    for v, g in gtv.items():
        if g["level"] == 1:
            out[v] = [{"class_name": g["classes"][0], "start_time_sec": None,
                       "end_time_sec": None}] if g["is_anomaly"] else []
            continue
        ev = []
        for c, s, e in g["segments"]:
            step = (e - s) / 5
            ev += [{"class_name": c, "start_time_sec": s + i * step,
                    "end_time_sec": s + (i + 1) * step} for i in range(5)]
        out[v] = ev
    return out


for name, fn in [("oracle (perfect)", oracle), ("empty (all normal)", dict),
                 ("whole-clip accident", whole_clip), ("spam all classes", spam_classes),
                 ("fragmented oracle", fragmented)]:
    r = score(gt, fn())
    print(f"{name:22s} L1 {r['level1']:.3f}  L2 {r['level2']:.3f}  "
          f"L3 {r['level3']:.3f}  overall {r['overall_mean']:.3f}")
