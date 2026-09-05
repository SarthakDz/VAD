"""Dataset loading for the AHC pack.

IMPORTANT -- train and test do NOT share a schema. Verified against the real
files on 2026-09-05:

    train/<class>/ground_truth.csv
        video_id,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary
    test/ground_truth.csv
        video_id,level,is_anomaly,class_name,start_time_sec,end_time_sec,description_summary

Train has no `level` column. The PRD's claim that both follow the same
two-file pattern is wrong; reality wins.

Other verified facts this module relies on:
  * train is one row per video (3173 rows, 3173 unique video_ids)
  * every train anomaly row has timestamps (2200/2200) -- no MIL needed
  * test is multi-row per video (52 rows, 34 videos)
  * description_summary is never blank in train, sometimes blank in test
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .labels import CLASSES, NORMAL, validate_series

TRAIN_COLS = [
    "video_id",
    "is_anomaly",
    "class_name",
    "start_time_sec",
    "end_time_sec",
    "description_summary",
]
TEST_COLS = [
    "video_id",
    "level",
    "is_anomaly",
    "class_name",
    "start_time_sec",
    "end_time_sec",
    "description_summary",
]
SUBMISSION_COLS = TEST_COLS


def _read_gt(path: Path, expected: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"video_id": str, "class_name": str, "description_summary": str},
        keep_default_na=False,
        na_values=[],
    )
    if list(df.columns) != expected:
        raise ValueError(
            f"{path}: unexpected columns.\n  got:      {list(df.columns)}\n  expected: {expected}"
        )
    # Empty strings -> NaN for the numeric interval columns only.
    for c in ("start_time_sec", "end_time_sec"):
        df[c] = pd.to_numeric(df[c].replace("", None), errors="coerce")
    df["is_anomaly"] = df["is_anomaly"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if df["is_anomaly"].isna().any():
        raise ValueError(f"{path}: unparseable is_anomaly values")
    bad = validate_series(df["class_name"])
    if bad:
        raise ValueError(f"{path}: class_name values not in CLASSES: {bad}")
    return df


def _read_videos_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    if list(df.columns) != ["video_id", "filename"]:
        raise ValueError(f"{path}: expected [video_id, filename], got {list(df.columns)}")
    return df


def load_train(root: str | Path) -> pd.DataFrame:
    """Concatenate all 12 class folders into one frame.

    Adds `source_class` (the folder it came from -- useful for source-aware
    splits, since several classes look like single-benchmark rips) and `path`.
    """
    root = Path(root)
    frames = []
    for class_dir in sorted((root / "train").iterdir()):
        if not class_dir.is_dir():
            continue
        gt = _read_gt(class_dir / "ground_truth.csv", TRAIN_COLS)
        vids = _read_videos_csv(class_dir / "videos.csv")
        merged = gt.merge(vids, on="video_id", how="left", validate="one_to_one")
        if merged["filename"].isna().any():
            missing = merged.loc[merged["filename"].isna(), "video_id"].tolist()
            raise ValueError(f"{class_dir}: video_ids absent from videos.csv: {missing[:5]}")
        merged["source_class"] = class_dir.name
        merged["path"] = merged["filename"].map(lambda f: str(class_dir / f))
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True)
    if df["video_id"].duplicated().any():
        raise ValueError("train video_ids are not unique across class folders")
    return df


def load_test(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    test_dir = root / "test"
    gt = _read_gt(test_dir / "ground_truth.csv", TEST_COLS)
    vids = _read_videos_csv(test_dir / "videos.csv")
    merged = gt.merge(vids, on="video_id", how="left", validate="many_to_one")
    if merged["filename"].isna().any():
        missing = sorted(set(merged.loc[merged["filename"].isna(), "video_id"]))
        raise ValueError(f"test: video_ids absent from videos.csv: {missing}")
    merged["path"] = merged["filename"].map(lambda f: str(test_dir / f))
    return merged


def load_test_videos(root: str | Path) -> pd.DataFrame:
    """Every test video with its level -- the row set a submission must cover.

    `level` lives only in ground_truth.csv, not videos.csv. On the private set
    the ground truth is withheld, so this mapping may not exist there. See the
    open question in the PRD; until it is answered, treat level as optional and
    always emit timestamps for anomalies.
    """
    root = Path(root)
    vids = _read_videos_csv(root / "test" / "videos.csv")
    gt = _read_gt(root / "test" / "ground_truth.csv", TEST_COLS)
    levels = gt.groupby("video_id")["level"].first()
    vids["level"] = vids["video_id"].map(levels).astype("Int64")
    vids["path"] = vids["filename"].map(lambda f: str(root / "test" / f))
    return vids


def missing_files(df: pd.DataFrame) -> list[str]:
    return [p for p in df["path"].unique() if not Path(p).exists()]


def summarise(df: pd.DataFrame, name: str) -> str:
    lines = [f"--- {name}: {len(df)} rows, {df['video_id'].nunique()} videos ---"]
    counts = df["class_name"].value_counts()
    for c in CLASSES:
        if c in counts:
            lines.append(f"  {c:34s} {counts[c]:5d}")
    n_ts = df["start_time_sec"].notna().sum()
    n_desc = (df["description_summary"].fillna("").str.len() > 0).sum()
    lines.append(f"  rows with timestamps:  {n_ts}/{len(df)}")
    lines.append(f"  rows with description: {n_desc}/{len(df)}")
    if "level" in df.columns:
        lines.append(f"  levels: {dict(df['level'].value_counts().sort_index())}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "../Train and Test"
    tr, te = load_train(root), load_test(root)
    print(summarise(tr, "train"))
    print(summarise(te, "test"))
    for label, d in (("train", tr), ("test", te)):
        miss = missing_files(d)
        print(f"{label}: {len(miss)} missing files" + (f" -- {miss[:3]}" if miss else ""))
