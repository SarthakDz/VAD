"""Encode a class-balanced subset.

`encode.run` sorts longest-first so big files surface problems early, which is
right for a full pass but wrong for a partial one: file size correlates with
class, so a truncated run yields 170 loitering clips and one fire clip. For a
deadline-bounded encode we want coverage, so take N per class instead.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.encode import run
from src.io_dataset import load_train

N = int(sys.argv[1]) if len(sys.argv) > 1 else 130
CACHE = Path(sys.argv[2] if len(sys.argv) > 2 else "cache_so400m")
df = load_train("../Train and Test")
items = []
for cls, g in df.groupby("class_name"):
    g = g.sort_values("video_id")           # deterministic, not size-ordered
    items += list(zip(g["video_id"].astype(str)[:N], g["path"].astype(str)[:N]))
print(f"{len(items)} clips, up to {N} per class")
run(items, CACHE, "google/siglip-so400m-patch14-384", 2.0, 16, 8,
    "cuda" if torch.cuda.is_available() else "cpu", resize=384)
