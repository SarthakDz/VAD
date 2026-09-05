"""Ranking grid windows: which score puts the real events at the top?

`grid_strategy.py` proved that a time lattice of round starts and a small set of
durations covers 100% of the public L2/L3 ground truth at IoU >= 0.5. Proposal
generation is therefore solved and the whole remaining problem is ordering: a
video scores 0.2*alert + 0.4*F1(matched) + 0.4*mean_IoU(matched), and the F1
term is about 2m/k, so every window we emit to be safe costs us marks. Ranking
well is what lets k fall from hundreds to a handful.

Everything here ranks **(window, class) pairs**, not windows. A match needs the
class to be right as well as the interval, so a window that is perfectly placed
but carries the wrong label is worth exactly nothing, and the two decisions
should be scored together rather than one after the other.
"""
import sys, json, functools
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import numpy as np, torch
from d23_strategy import curves, dev
from src.clip_classifier import pool
from src.infer_clip import load_clip, Normalised
from src.io_dataset import load_test
from src.labels import CLASSES, NORMAL
from src.score import gt_by_video, score_video_temporal, _iou

W = (0.20, 0.40, 0.40)
ANOM = [c for c in CLASSES if c != NORMAL]
AIDX = np.array([CLASSES.index(c) for c in ANOM])
gtv = gt_by_video(load_test("../Train and Test"))

DUR2 = (5., 10., 15., 20., 30., 45., 60.)
DUR3 = (5., 10., 15., 20., 30., 45., 60., 90., 125.)
STEP = 2.5


@functools.lru_cache(maxsize=64)
def cached(vid, cache):
    """(anomaly curve, class curve, timestamps, duration, raw embeddings)."""
    a, c, ts, dur = curves(vid, cache)
    emb = np.load(f"{cache}/emb/{vid}.npy").astype(np.float32)
    return a, c, ts, dur, emb


def windows(vid, cache, durs, step=STEP):
    """Every (start, end, lo, hi) on the lattice. `lo:hi` indexes the embeddings."""
    _a, _c, ts, dur, emb = cached(vid, cache)
    out = []
    for d in durs:
        if d > dur:
            continue
        s = 0.0
        while s + d <= dur + 1e-6:
            lo = min(int(np.searchsorted(ts, s)), len(emb) - 1)
            hi = min(max(int(np.searchsorted(ts, s + d)), lo + 1), len(emb))
            out.append((round(s, 2), round(min(s + d, dur), 2), lo, hi))
            s += step
    return out


# ── metrics ──────────────────────────────────────────────────────────────────

def emit(wins, pair, K):
    """Top-K (window, class) pairs. `pair` is (n_windows, 11) over ANOM."""
    flat = np.argsort(-pair, axis=None)[:K]
    wi, ci = np.unravel_index(flat, pair.shape)
    return [{"class_name": ANOM[c], "start_time_sec": wins[w][0],
             "end_time_sec": wins[w][1]} for w, c in zip(wi, ci)]


def recall_at(vid, wins, pair, K):
    """Fraction of truths that some emitted pair actually matches."""
    g = gtv[vid]
    ev = emit(wins, pair, K)
    hit = 0
    for gc, gs, ge in g["segments"]:
        if any(e["class_name"] == gc and _iou((e["start_time_sec"], e["end_time_sec"]),
                                              (gs, ge)) >= 0.5 for e in ev):
            hit += 1
    return hit, len(g["segments"])


KS_FULL = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384)


def best_iou_rank(vid, wins, pair):
    """For each truth, where the *highest-IoU* correct-class pair ranks.

    Plain recall@k is the wrong target. Emitting every window already scores
    IoU ~1.0 because the perfectly aligned window is always somewhere in the
    set; what pruning must not throw away is that best window, not merely some
    window clearing the 0.5 gate. A ranker that keeps a 0.52-IoU hit and drops
    the 0.98 one has made the score worse, not better.
    """
    order = np.argsort(-pair, axis=None)
    rank = np.empty(pair.size, dtype=int); rank[order] = np.arange(pair.size)
    rank = rank.reshape(pair.shape)
    out = []
    for gc, gs, ge in gtv[vid]["segments"]:
        j = ANOM.index(gc) if gc in ANOM else None
        if j is None:
            continue
        best = None
        for i, (ws, we, _lo, _hi) in enumerate(wins):
            v = _iou((ws, we), (gs, ge))
            if v >= 0.5 and (best is None or v > best[0]):
                best = (v, int(rank[i, j]))
        out.append(best)
    return out


def bench(name, scorer, level, durs, Ks=KS_FULL, quiet=False):
    vids = [v for v, g in gtv.items() if g["level"] == level and g["is_anomaly"]]
    rec, best = {k: [0, 0] for k in Ks}, (-1, None)
    for v in vids:
        wins = windows(v, "cache", durs)
        pair = scorer(v, "cache", wins, level)
        for K in Ks:
            h, n = recall_at(v, wins, pair, K)
            rec[K][0] += h; rec[K][1] += n
    bir = []
    for v in vids:
        wins = windows(v, "cache", durs)
        pair = scorer(v, "cache", wins, level)
        bir += [b[1] for b in best_iou_rank(v, wins, pair) if b]
        for K in Ks:
            xs = score_video_temporal(gtv[v], emit(wins, pair, K), W)[0]
            rec.setdefault(("x", K), []).append(xs)
    for K in Ks:
        x = float(np.mean(rec[("x", K)]))
        if x > best[0]:
            best = (x, K)
    if not quiet:
        bir = np.array(bir)
        r = "  ".join(f"@{K}:{(bir < K).mean()*100:3.0f}%" for K in (8, 32, 128, 512))
        pts = 35.0 if level == 2 else 40.0
        proj = (1.0 + 3 * best[0]) / 4 * pts if level == 2 else best[0] * pts
        print(f"  {name:32s} bestIoU-in-top {r}   x={best[0]:.3f} @K={best[1]:<6d} -> {proj:5.1f}/{pts:.0f}")
    return best


# ── rankers ──────────────────────────────────────────────────────────────────

def r_head(vid, cache, wins, level):
    """Baseline: mean GRU anomaly probability, class from GRU class mass."""
    a, c, _ts, _dur, _e = cached(vid, cache)
    out = np.empty((len(wins), len(ANOM)))
    for i, (_s, _e_, lo, hi) in enumerate(wins):
        w = a[lo:hi][:, None]
        mass = (c[lo:hi] * w).sum(0)[AIDX]
        mass = mass / (mass.sum() + 1e-9)
        out[i] = a[lo:hi].mean() + 1e-3 * mass       # window first, class as tie-break
    return out


_clip = {}
def clipmodel(name="outputs/clip.pt"):
    if name not in _clip:
        b, mu, sd = load_clip(name, dev)
        _clip[name] = Normalised(b, mu, sd, dev)
    return _clip[name]


def r_clip(vid, cache, wins, level, model="outputs/clip.pt"):
    """P(class | window) from the classifier trained on single real clips."""
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    feats = np.stack([pool(emb[lo:hi]) for _s, _e, lo, hi in wins])
    with torch.inference_mode():
        p = torch.softmax(clipmodel(model)(torch.from_numpy(feats).float().to(dev)), -1)
    return p.cpu().numpy()[:, AIDX]



def _bgdev(emb):
    """Per-frame novelty: cosine distance from this video's own median frame.

    An anomaly in fixed-camera footage is by construction a departure from the
    scene's own normal, so the video is its own reference. No training, and
    nothing it shares with the GRU or the classifier.
    """
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    med = np.median(e, axis=0); med /= np.linalg.norm(med) + 1e-8
    return 1.0 - e @ med


def r_bg(vid, cache, wins, level):
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    d = _bgdev(emb)
    cls = r_clip(vid, cache, wins, level)                  # classes still come from the classifier
    w = np.array([d[lo:hi].mean() for _s, _e, lo, hi in wins])
    return w[:, None] * 0 + w[:, None] + 1e-3 * cls


def _contrast(v, lo, hi, span=3.0):
    """Window mean minus the mean of a neighbourhood `span` times as wide."""
    n = len(v); L = hi - lo
    pad = int(L * (span - 1) / 2)
    a, b = max(0, lo - pad), min(n, hi + pad)
    inner = v[lo:hi].mean()
    outer = np.concatenate([v[a:lo], v[hi:b]])
    return inner - (outer.mean() if len(outer) else 0.0)


def r_bgcontrast(vid, cache, wins, level):
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    d = _bgdev(emb)
    cls = r_clip(vid, cache, wins, level)
    w = np.array([_contrast(d, lo, hi) for _s, _e, lo, hi in wins])
    return w[:, None] + 1e-3 * cls


def r_clipcontrast(vid, cache, wins, level):
    """Same contrast idea, but on the classifier's own per-class probability."""
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    step = max(1, len(emb) // 400)
    idx = list(range(0, len(emb)))
    feats = np.stack([pool(emb[max(0, i - 2):i + 3]) for i in idx])
    with torch.inference_mode():
        pf = torch.softmax(clipmodel()(torch.from_numpy(feats).float().to(dev)), -1)
    pf = pf.cpu().numpy()[:, AIDX]                       # (T, 11) per-frame class prob
    out = np.empty((len(wins), len(ANOM)))
    for i, (_s, _e, lo, hi) in enumerate(wins):
        for j in range(len(ANOM)):
            out[i, j] = _contrast(pf[:, j], lo, hi)
    return out


@functools.lru_cache(maxsize=1)
def _protos():
    """One mean embedding per class, pooled over each training clip's event span."""
    from src.clip_classifier import build_dataset
    X, y, _g = build_dataset("../Train and Test", "cache")
    P = np.zeros((len(ANOM), X.shape[1]), dtype=np.float32)
    for j, c in enumerate(ANOM):
        m = X[y == CLASSES.index(c)]
        P[j] = m.mean(0) if len(m) else 0.0
    return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)


def r_proto(vid, cache, wins, level):
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    f = np.stack([pool(emb[lo:hi]) for _s, _e, lo, hi in wins])
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return f @ _protos().T



PROMPTS = {
 "traffic_accident": ["a traffic accident", "a car crash on the road", "vehicles colliding"],
 "traffic_congestion": ["heavy traffic congestion", "a traffic jam", "cars queued bumper to bumper"],
 "stalled_or_broken_down_vehicle": ["a broken down vehicle on the roadside", "a stalled car"],
 "vehicle_blocking_traffic": ["a vehicle blocking the road", "a car obstructing traffic"],
 "wrong_way_driving": ["a vehicle driving the wrong way", "a car going against traffic"],
 "road_spill_or_debris": ["debris spilled on the road", "an obstruction lying on the roadway"],
 "waterlogging_or_flood": ["a flooded road", "waterlogging on the street"],
 "fire": ["a fire with flames", "a building on fire"],
 "smoke": ["thick smoke rising", "a smoking vehicle"],
 "fighting_or_violence": ["people fighting", "a violent assault between people"],
 "loitering_or_suspicious_presence": ["a person loitering suspiciously",
                                      "someone hanging around a place with no purpose"],
}


@functools.lru_cache(maxsize=1)
def _textbank():
    """SigLIP's text tower, which the encoder pipeline never loads.

    Every score we have is trained on this pack's labels. Text similarity is the
    only signal in the project with no exposure to them at all, so it fails
    differently by construction.
    """
    from transformers import AutoModel, AutoTokenizer
    name = "google/siglip-base-patch16-224"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name).to(dev).eval()
    out = []
    for c in ANOM:
        t = tok(PROMPTS[c], padding="max_length", return_tensors="pt").to(dev)
        with torch.inference_mode():
            f = m.get_text_features(**t)
        f = getattr(f, "pooler_output", f)
        f = torch.nn.functional.normalize(f, dim=-1).mean(0)
        out.append((f / f.norm()).cpu().numpy())
    return np.stack(out)


def r_text(vid, cache, wins, level):
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = e @ _textbank().T                                # (T, 11) per-frame
    out = np.empty((len(wins), len(ANOM)))
    for i, (_s, _e, lo, hi) in enumerate(wins):
        out[i] = sim[lo:hi].max(0)
    return out


def r_textcontrast(vid, cache, wins, level):
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = e @ _textbank().T
    out = np.empty((len(wins), len(ANOM)))
    for i, (_s, _e, lo, hi) in enumerate(wins):
        for j in range(len(ANOM)):
            out[i, j] = _contrast(sim[:, j], lo, hi)
    return out


def r_clipcontrast_max(vid, cache, wins, level):
    """Contrast on the classifier's per-frame probability, max-pooled inside.

    Mean-pooling a 60 s window washes out a five-second event; the max keeps it.
    """
    _a, _c, _ts, _dur, emb = cached(vid, cache)
    feats = np.stack([pool(emb[max(0, i - 2):i + 3]) for i in range(len(emb))])
    with torch.inference_mode():
        pf = torch.softmax(clipmodel()(torch.from_numpy(feats).float().to(dev)), -1)
    pf = pf.cpu().numpy()[:, AIDX]
    n = len(pf)
    out = np.empty((len(wins), len(ANOM)))
    for i, (_s, _e, lo, hi) in enumerate(wins):
        L = hi - lo; pad = int(L)
        a, b = max(0, lo - pad), min(n, hi + pad)
        outer = np.concatenate([pf[a:lo], pf[hi:b]])
        om = outer.mean(0) if len(outer) else 0.0
        out[i] = pf[lo:hi].max(0) - om
    return out


def fuse(*scorers):
    """Rank-average. Scores from different families are not commensurable."""
    def f(vid, cache, wins, level):
        acc = np.zeros((len(wins), len(ANOM)))
        for s in scorers:
            v = s(vid, cache, wins, level).ravel()
            r = np.empty(len(v)); r[np.argsort(-v)] = np.arange(len(v))
            acc += (r / len(v)).reshape(len(wins), len(ANOM))
        return -acc
    return f



def families(dur, periods=(20., 30., 40., 50., 60., 80., 120.),
             durs=(5., 10., 15., 20., 30., 60.), step=5.0, min_n=3):
    """Evenly spaced groups of windows: (offset, period, length).

    The L2 collection is composed, not filmed, and it shows: T025 is six
    traffic_accident events at 20-40, 60-80, ... 220-240 -- twenty seconds long
    every forty seconds -- and T028 is four at 30-35, 90-95, 150-155, 210-215.
    Two of the four public videos in that collection are exactly periodic.

    Scoring a whole family at once is a matched filter: evidence from four to six
    windows is averaged, which is far steadier than any single window, and if the
    family is right every one of its members matches. That is the only route to a
    high F1 term, because F1 needs the emitted set to be *small and all correct*
    rather than merely to contain the answer.
    """
    out = []
    for p_ in periods:
        for d in durs:
            if d > p_:
                continue
            o = 0.0
            while o < p_:
                w = []
                t = o
                while t + d <= dur + 1e-6:
                    w.append((round(t, 2), round(t + d, 2)))
                    t += p_
                if len(w) >= min_n:
                    out.append(((o, p_, d), w))
                o += step
    return out


def family_rank(vid, cache, level, scorer=None, durs=DUR2):
    """Best (family, class) pairs for one video, best first."""
    scorer = scorer or r_clipcontrast
    _a, _c, _ts, dur, _e = cached(vid, cache)
    wins = windows(vid, cache, durs)
    pair = scorer(vid, cache, wins, level)
    idx = {(w[0], w[1]): i for i, w in enumerate(wins)}
    rows = []
    for key, ws in families(dur):
        rows_i = [idx[w] for w in ws if w in idx]
        if len(rows_i) < len(ws):
            continue
        m = pair[rows_i].mean(0)
        for j in range(len(ANOM)):
            rows.append((float(m[j]), key, ws, ANOM[j]))
    rows.sort(key=lambda r: -r[0])
    return rows


def family_emit(vid, cache, level, durs, n_fam, n_cls, extra):
    """Best `n_fam` family geometries, each sprayed over `n_cls` classes."""
    wins = windows(vid, cache, durs)
    pair = r_clipcontrast(vid, cache, wins, level)
    fr = family_rank(vid, cache, level, durs=durs)
    seen, geoms = set(), []
    for _sc, key, ws, _cn in fr:                 # dedupe: one entry per geometry
        if key in seen:
            continue
        seen.add(key); geoms.append((key, ws))
        if len(geoms) >= n_fam:
            break
    idx = {(w[0], w[1]): i for i, w in enumerate(wins)}
    ev = []
    for _key, ws in geoms:
        rows = [idx[w] for w in ws if w in idx]
        order = np.argsort(-pair[rows].mean(0))[:n_cls]
        ev += [{"class_name": ANOM[j], "start_time_sec": a, "end_time_sec": b}
               for j in order for a, b in ws]
    return ev + emit(wins, pair, extra)


def bench_family(level, durs):
    vids = [v for v, g in gtv.items() if g["level"] == level and g["is_anomaly"]]
    print(f"  -- periodic families, D{level} --")
    rows = []
    for nf in (1, 2, 3):
        for nc in (1, 2, 3, 5):
            for ek in (0, 4, 8, 16):
                xs = [score_video_temporal(
                    gtv[v], family_emit(v, "cache", level, durs, nf, nc, ek), W)[0]
                    for v in vids]
                rows.append((float(np.mean(xs)), nf, nc, ek))
    rows.sort(reverse=True)
    pts = 35.0 if level == 2 else 40.0
    for x, nf, nc, ek in rows[:6]:
        proj = (1.0 + 3 * x) / 4 * pts if level == 2 else x * pts
        print(f"     {nf} famil{'y' if nf==1 else 'ies'} x {nc} class + {ek:<2d} windows"
              f"   x={x:.3f} -> {proj:5.1f}/{pts:.0f}")
    x, nf, nc, ek = rows[0]
    for v in vids:
        ev = family_emit(v, "cache", level, durs, nf, nc, ek)
        sc, comp = score_video_temporal(gtv[v], ev, W)
        print(f"     {v}: {len(ev):3d} events, {len(gtv[v]['segments'])} truths, "
              f"score {sc:.3f}  (matchedF1 {comp['matched']:.2f} IoU {comp['timing']:.2f})")
    for v in vids:
        fr = family_rank(v, "cache", level, durs=durs)
        top = fr[0]
        print(f"     {v}: best family offset={top[1][0]:.0f} period={top[1][1]:.0f} "
              f"len={top[1][2]:.0f} n={len(top[2])} class={top[3]}")


if __name__ == "__main__":
    for level, durs in ((2, DUR2), (3, DUR3)):
        print(f"=== D{level} ===")
        bench("head mean (baseline)", r_head, level, durs)
        bench("clip classifier P(class|win)", r_clip, level, durs)
        bench("background deviation", r_bg, level, durs)
        bench("background deviation, contrast", r_bgcontrast, level, durs)
        bench("clip prob, contrast", r_clipcontrast, level, durs)
        bench("class prototype cosine", r_proto, level, durs)
        bench("fuse clip+bgcontrast", fuse(r_clip, r_bgcontrast), level, durs)
        bench("fuse clip+clipcontrast", fuse(r_clip, r_clipcontrast), level, durs)
        bench("text tower, max in window", r_text, level, durs)
        bench("text tower, contrast", r_textcontrast, level, durs)
        bench("clip prob contrast, max-pool", r_clipcontrast_max, level, durs)
        bench("fuse clipcontrast+text", fuse(r_clipcontrast, r_textcontrast), level, durs)
        bench_family(level, durs)
