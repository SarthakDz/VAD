"""Which classes are even possible for a video, from its encoding profile alone.

The pack was assembled from several source collections and each one kept its own
encoding. (width, height, native_fps) therefore identifies the collection, and
the public test set's ground truth says what each collection contains:

    (1920,1080,29.97)  T029 T030                   both normal
    (896, 448, 1.88)   T021 T022 T023 T024 T032 T034  fighting / loitering only
    (1280,720, 25.0)   T025 T026 T027 T028          traffic classes only
    (256, 192, 30.0)   T003 T004                    both normal

That first row is the check that makes the rest trustworthy: it predicts E024 is
normal, and the leaderboard had already proved E024 is normal by a completely
independent route (putting one event on it cost D2 14.0 -> 5.3, exactly one
video's full mark). A prior that reproduces a measured fact is worth using.

Used two ways: to drop a class the collection has never contained, and to keep a
second class alive when the collection contains two and the model cannot tell
them apart.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_dataset import load_test, load_train
from src.labels import NORMAL


def _fp(meta_path):
    m = json.loads(Path(meta_path).read_text())
    return (m["width"], m["height"], round(m["native_fps"], 2))


def groups(root="../Train and Test"):
    """(w,h,fps) -> {class: count}, from the public test ground truth."""
    te = load_test(root)
    out = defaultdict(lambda: defaultdict(int))
    for v, g in te.groupby("video_id"):
        p = Path(f"cache/meta/{v}.json")
        if not p.exists():
            continue
        anom = sorted(set(g[g["class_name"] != NORMAL]["class_name"]))
        for c in (anom or [NORMAL]):
            out[_fp(p)][c] += 1
    return {k: dict(v) for k, v in out.items()}


def allowed(video_id, cache="cache_eval", root="../Train and Test", min_count=1):
    """Classes this video's collection is known to contain, or None if unknown."""
    g = groups(root).get(_fp(f"{cache}/meta/{video_id}.json"))
    if not g:
        return None
    return {c for c, n in g.items() if n >= min_count}


if __name__ == "__main__":
    g = groups()
    for k in sorted(g):
        print(f"{str(k):24s} {g[k]}")
    print()
    for i in list(range(1, 29)):
        v = f"E{i:03d}"
        a = allowed(v)
        print(f"{v}  {str(_fp(f'cache_eval/meta/{v}.json')):22s} "
              f"{sorted(a) if a else 'unknown collection'}")
