"""Sweep for the leaderboard objective: precision and false-alarm suppression."""
import itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_dataset import TEST_COLS, _read_gt
from src.leaderboard import report
from src.score import score as pvscore
from src.submit import manifest_from_public_test
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import load_all, predict  # noqa

ROOT = "../Train and Test"
mf = manifest_from_public_test(ROOT)
gt = _read_gt(Path(ROOT) / "test" / "ground_truth.csv", TEST_COLS)
curves = load_all(Path("cache"), mf)

grid = list(itertools.product(
    [0.6, 0.75, 0.85, 0.92],        # enter
    [0.3, 0.5, 0.7],                # exit
    [10, 20, 40],                   # merge gap
    [1, 3, 8],                      # min event sec
    [0, 1, 2, 3],                   # top_k per video (0 = unlimited)
    [0.0, 0.8, 0.9],                # min mean score per segment
))
rows = []
for en, ex, gp, mn, tk, ms in grid:
    if ex >= en:
        continue
    pr = predict(curves, mf, en, ex, gp, mn, tk, ms)
    r = report(gt, pr)
    pv = pvscore(gt, pr)
    d = {x["level"]: x for x in r["difficulties"]}
    rows.append({"enter": en, "exit": ex, "gap": gp, "min_ev": mn, "top_k": tk,
                 "min_score": ms, "total": r["total_proxy"],
                 "marks": pv["marks"], "l1": pv["level1"], "l2": pv["level2"], "l3": pv["level3"],
                 "fa": sum(x["fa"] for x in r["difficulties"]),
                 "found": sum(x["found"] for x in r["difficulties"]),
                 **{f"d{k}_p": v["precision"] for k, v in d.items()},
                 **{f"d{k}_m": v["marks_proxy"] for k, v in d.items()}})
rows.sort(key=lambda r: (round(r["marks"],1), -r["fa"]), reverse=True)
print(f"{len(rows)} configs\n")
h = f"{'ent':>5}{'exi':>5}{'gap':>5}{'mev':>4}{'topk':>5}{'msc':>5} | {'found':>6}{'FA':>5} | {'L1':>6}{'L2':>6}{'L3':>6}{'MARKS':>8}"
print(h); print("-"*len(h))
for r in rows[:18]:
    print(f"{r['enter']:5.2f}{r['exit']:5.2f}{r['gap']:5.0f}{r['min_ev']:4.0f}{r['top_k']:5d}{r['min_score']:5.2f} | "
          f"{r['found']:6d}{r['fa']:5d} | {r['l1']:6.3f}{r['l2']:6.3f}{r['l3']:6.3f}{r['marks']:8.1f}")
b = rows[0]
print(f"\nbest: --enter {b['enter']} --exit {b['exit']} --merge-gap {b['gap']} "
      f"--min-event {b['min_ev']} --top-k {b['top_k']} --min-score {b['min_score']}")
json.dump(rows[:40], open("outputs/sweep_lb.json", "w"), indent=1)
