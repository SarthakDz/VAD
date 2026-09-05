"""Search for the arithmetic progression the events were composed on.

Two of the four anomalous public L2 videos have ground truth that is exactly
periodic: T025 is six traffic_accident events at 20+40i, twenty seconds long,
and T028 is four at 30+60i, five seconds long. T027 and T026 are not periodic
but every boundary still lands on a multiple of five.

That suggests a different search than ranking windows one at a time. Instead of
asking "is this window anomalous", enumerate whole hypotheses -- n events of
length d starting at a and repeating every b -- and ask which hypothesis the
anomaly curve supports. Averaging the curve over four to six windows is far less
noisy than reading it at one, which is exactly the regime our head is bad in:
its recall@32 for individual true windows is 44% on L2 and 12% on L3.

Scored by contrast: mean anomaly inside the hypothesis minus mean outside. A
hypothesis that merely covers a lot of video gets no credit for it.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np
from d23_strategy import curves
from src.io_dataset import load_test
from src.labels import CLASSES, NORMAL
from src.score import gt_by_video, _iou

GRID = 5.0
DURS = (5., 10., 15., 20., 25., 30., 45., 60.)
NS = range(2, 9)


def hypotheses(dur, min_spread=0.5):
    """Non-overlapping progressions that span at least `min_spread` of the video.

    Both constraints are there to stop the search degenerating. Without
    `b >= d` the best-scoring hypothesis on E022 was eight windows overlapping
    each other at a five-second period, which is a single spike wearing a
    progression's clothes. Without the spread constraint the search happily
    packs every event into whichever twenty seconds the curve likes best. Real
    compositions look like T025 (b=40, d=20, spanning 220 of 240 s).
    """
    out = []
    for d in DURS:
        for b in np.arange(GRID, dur, GRID):
            if b < d:                      # events must not overlap each other
                continue
            for n in NS:
                span = (n - 1) * b + d
                if span > dur + 1e-6:
                    break
                if span < min_spread * dur:
                    continue
                for a in np.arange(0.0, dur - span + GRID, GRID):
                    if a + span > dur + 1e-6:
                        break
                    out.append((n, float(a), float(b), float(d)))
    return out


def windows(h):
    n, a, b, d = h
    return [(a + i * b, a + i * b + d) for i in range(n)]


def score_h(h, a_curve, ts, dur):
    inside = np.zeros(len(a_curve), bool)
    for s, e in windows(h):
        inside[(ts >= s) & (ts < e)] = True
    if inside.sum() == 0 or (~inside).sum() == 0:
        return -1e9
    return float(a_curve[inside].mean() - a_curve[~inside].mean())


def best_class(h, a_curve, c_curve, ts):
    inside = np.zeros(len(a_curve), bool)
    for s, e in windows(h):
        inside[(ts >= s) & (ts < e)] = True
    mass = (c_curve[inside] * a_curve[inside][:, None]).sum(0)
    mass[CLASSES.index(NORMAL)] = -np.inf
    return [CLASSES[i] for i in np.argsort(-mass)]


def rank(vid, cache="cache", top=50):
    a, c, ts, dur = curves(vid, cache)
    hs = hypotheses(dur)
    scored = sorted(((score_h(h, a, ts, dur), h) for h in hs), reverse=True)
    return [(s, h, best_class(h, a, c, ts)) for s, h in scored[:top]], dur


if __name__ == "__main__":
    gtv = gt_by_video(load_test("../Train and Test"))
    for lvl in (2, 3):
        print(f"=== public L{lvl} ===")
        for v, g in sorted(gtv.items()):
            if g["level"] != lvl or not g["is_anomaly"]:
                continue
            top, dur = rank(v, top=400)
            truth = g["segments"]
            # where does the first hypothesis that matches >=half the truth rank?
            best_rank, best_hits = None, 0
            for r, (s, h, order) in enumerate(top):
                ws = windows(h)
                hits = sum(1 for gc, gs, ge in truth
                           if any(_iou(w, (gs, ge)) >= 0.5 for w in ws))
                if hits > best_hits or (hits == best_hits and best_rank is None):
                    if hits > best_hits:
                        best_hits, best_rank = hits, r
            s, h, order = top[0]
            n, a, b, d = h
            print(f"  {v}  dur {dur:5.0f}s  truth {len(truth)} events "
                  f"({', '.join(f'{gs:.0f}-{ge:.0f}' for _, gs, ge in truth[:6])})")
            print(f"     top hypothesis: n={n} a={a:.0f} b={b:.0f} d={d:.0f} "
                  f"contrast {s:+.3f} class {order[0]}")
            print(f"     best hypothesis in top 400 matches {best_hits}/{len(truth)} "
                  f"truths, at rank {best_rank}")
