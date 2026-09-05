"""Encode the private evaluation set and write its manifest.

The arena switched from the public T0xx set to this private E0xx set, which is
why a T0xx submission now returns "no videos in this file belong to this level".
Level comes from the directory (L1/L2/L3) rather than a manifest field.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.encode import run

ROOT = Path("../Evaluation")
items, levels = [], {}
for lvl in (1, 2, 3):
    d = ROOT / f"L{lvl}"
    for line in (d / "videos.csv").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        vid, fn = line.strip().split(",")
        items.append((vid, str(d / fn)))
        levels[vid] = lvl
print(f"{len(items)} eval videos: " + ", ".join(f"L{l}={sum(1 for v in levels.values() if v==l)}" for l in (1,2,3)))

which = sys.argv[1] if len(sys.argv) > 1 else "base"
dev = "cuda" if torch.cuda.is_available() else "cpu"
if which == "base":
    run(items, Path("cache_eval"), "google/siglip-base-patch16-224", 2.0, 64, 4096, dev, resize=224)
else:  # so400m, L1 only -- the ensemble D1 classifier needs it, D2/D3 use the head
    l1 = [(v, p) for v, p in items if levels[v] == 1]
    run(l1, Path("cache_eval_so"), "google/siglip-so400m-patch14-384", 2.0, 16, 8, dev, resize=384)

Path("data").mkdir(exist_ok=True)
Path("data/manifest_eval.json").write_text(json.dumps(
    {"schema_version": "1.0",
     "videos": [{"video_id": v, "level": levels[v], "domain": "", "duration_sec": 0}
                for v, _ in items]}, indent=1))
print("wrote data/manifest_eval.json")
