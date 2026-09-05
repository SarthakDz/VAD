"""Submission construction and validation.

Nothing reaches disk without passing `validate`. A malformed submission is a
zero; a wrong prediction is merely a wrong prediction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_dataset import SUBMISSION_COLS, load_test_videos
from .labels import NORMAL, validate_series


class SubmissionError(ValueError):
    pass


def validate(
    df: pd.DataFrame,
    required_video_ids: list[str],
    durations: dict[str, float] | None = None,
) -> None:
    """Raise SubmissionError on anything the scorer could choke on."""
    problems: list[str] = []

    if list(df.columns) != SUBMISSION_COLS:
        problems.append(f"columns are {list(df.columns)}, expected {SUBMISSION_COLS}")

    bad = validate_series(df["class_name"].fillna(""))
    if bad:
        problems.append(f"class_name values not in CLASSES: {bad}")

    covered = set(df["video_id"])
    missing = [v for v in required_video_ids if v not in covered]
    if missing:
        problems.append(f"{len(missing)} test videos absent from submission: {missing[:10]}")

    extra = covered - set(required_video_ids)
    if extra:
        problems.append(f"video_ids not in the test set: {sorted(extra)[:10]}")

    for col in ("video_id", "is_anomaly", "class_name"):
        if df[col].isna().any():
            problems.append(f"{col} contains NaN")

    anom = df[df["class_name"] != NORMAL]
    ts = anom.dropna(subset=["start_time_sec", "end_time_sec"])
    inverted = ts[ts["start_time_sec"] >= ts["end_time_sec"]]
    if len(inverted):
        problems.append(
            f"{len(inverted)} rows with start_time_sec >= end_time_sec "
            f"(e.g. {inverted.iloc[0]['video_id']})"
        )
    if (ts["start_time_sec"] < 0).any():
        problems.append("negative start_time_sec")

    normal_rows = df[df["class_name"] == NORMAL]
    if normal_rows["start_time_sec"].notna().any():
        problems.append("normal rows must have empty timestamps")
    if normal_rows["is_anomaly"].any():
        problems.append("rows with class_name=normal must have is_anomaly=false")
    if (~anom["is_anomaly"]).any():
        problems.append("anomaly-class rows must have is_anomaly=true")

    if durations:
        over = ts[ts.apply(
            lambda r: r["end_time_sec"] > durations.get(r["video_id"], float("inf")) + 1.0,
            axis=1,
        )]
        if len(over):
            problems.append(f"{len(over)} rows with end_time_sec past the video duration")

    if problems:
        raise SubmissionError("submission failed validation:\n  - " + "\n  - ".join(problems))


def write(
    df: pd.DataFrame,
    out_path: str | Path,
    required_video_ids: list[str],
    durations: dict[str, float] | None = None,
) -> Path:
    df = df.reindex(columns=SUBMISSION_COLS)
    validate(df, required_video_ids, durations)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["is_anomaly"] = out["is_anomaly"].map({True: "true", False: "false"})
    out.to_csv(out_path, index=False, na_rep="")
    return out_path


def baseline(root: str | Path, class_name: str = NORMAL) -> pd.DataFrame:
    """One row per test video, all predicted `class_name`, no timestamps.

    The M0 sanity check: proves the harness round-trips before any model exists.
    """
    vids = load_test_videos(root)
    return pd.DataFrame(
        {
            "video_id": vids["video_id"],
            "level": vids["level"],
            "is_anomaly": class_name != NORMAL,
            "class_name": class_name,
            "start_time_sec": pd.NA,
            "end_time_sec": pd.NA,
            "description_summary": "",
        }
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Train and Test")
    ap.add_argument("--class-name", default=NORMAL)
    ap.add_argument("--out", default="outputs/baseline_normal.csv")
    a = ap.parse_args()

    vids = load_test_videos(a.root)
    df = baseline(a.root, a.class_name)
    p = write(df, a.out, vids["video_id"].tolist())
    print(f"wrote {p}  ({len(df)} rows, all '{a.class_name}') -- validation passed")
