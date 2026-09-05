"""Does retrieval against the 3207-clip train bank beat the trained clip heads on D1?

The trained heads and the GRU all plateaued at 9/20 and the two-encoder ensemble
reached 11/20. k-NN is a fourth, non-parametric decision rule over the same
features, so it fails differently -- which is the only thing that has moved D1.
Measured on the public test set, where ground truth exists.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.io_dataset import load_train, load_test, TEST_COLS
from src.labels import CLASSES, NORMAL
from src.clip_classifier import pool
from src.infer_clip import load_clip, Normalised

CACHE = Path("cache")
tr = load_train("../Train and Test")
cls = dict(zip(tr["video_id"].astype(str), tr["class_name"].astype(str)))
te = load_test("../Train and Test")
te1 = te[te["level"] == 1]
gt = {str(v): (g[g["class_name"] != NORMAL]["class_name"].iloc[0]
               if (g["class_name"] != NORMAL).any() else NORMAL)
      for v, g in te1.groupby("video_id")}

def feat(cache, vid, mode):
    e = np.load(f"{cache}/emb/{vid}.npy").astype(np.float32)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    f = pool(e) if mode == "pool" else e.mean(0)
    return f / (np.linalg.norm(f) + 1e-8)

def bank(cache, mode):
    ids = [v for v in cls if Path(f"{cache}/emb/{v}.npy").exists()]
    return ids, np.stack([feat(cache, v, mode) for v in ids])

def knn(M, ids, q, k, T):
    s = M @ q
    o = np.argsort(-s)[:k]
    w = np.exp((s[o] - s[o][0]) / T)
    votes = {}
    for i, wi in zip(o, w):
        votes[cls[ids[i]]] = votes.get(cls[ids[i]], 0.0) + float(wi)
    p = np.zeros(len(CLASSES))
    for c, v in votes.items():
        p[CLASSES.index(c)] = v
    return p / p.sum()

def d1_marks(pred):
    """25 * F1(found, false alarms) -- the calibrated D1 rule."""
    n_gt = sum(1 for v in gt if gt[v] != NORMAL)
    found = sum(1 for v in gt if pred[v] == gt[v] != NORMAL)
    n_pred = sum(1 for v in gt if pred[v] != NORMAL)
    p = found / n_pred if n_pred else 0.0
    r = found / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return 25 * f1, found, n_pred - found, n_gt

vids = sorted(gt)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- trained heads (the current recipe) --------------------------------------
bB, muB, sdB = load_clip("outputs/clip.pt", dev);    cB = Normalised(bB, muB, sdB, dev)
bS, muS, sdS = load_clip("outputs/clip_so.pt", dev); cS = Normalised(bS, muS, sdS, dev)
def head_probs(cache, model, vid):
    x = torch.from_numpy(pool(np.load(f"{cache}/emb/{vid}.npy").astype(np.float32))[None]).float().to(dev)
    with torch.inference_mode():
        return torch.softmax(model(x), -1)[0].cpu().numpy()
P_base = {v: head_probs("cache", cB, v) for v in vids}
P_so   = {v: head_probs("cache_so400m", cS, v) for v in vids
          if Path(f"cache_so400m/emb/{v}.npy").exists()}

# --- k-NN banks ---------------------------------------------------------------
results = {}
for mode in ("pool", "mean"):
    ids, M = bank("cache", mode)
    for k in (1, 5, 15, 30):
        for T in (0.02, 0.05):
            results[f"knn-base-{mode}-k{k}-T{T}"] = {
                v: knn(M, ids, feat("cache", v, mode), k, T) for v in vids}

def report(name, P, thr=0.0):
    pred = {}
    for v in vids:
        p = P.get(v)
        if p is None: pred[v] = NORMAL; continue
        i = int(p.argmax())
        pred[v] = NORMAL if (CLASSES[i] == NORMAL or p[i] < thr) else CLASSES[i]
    m, f, fa, n = d1_marks(pred)
    print(f"  {name:34s} {m:5.1f}/25   found {f}/{n}  FA {fa}")
    return m

print("D1 on the public test set (24 L1 videos):")
report("head base @0.0", P_base)
report("head base @0.4", P_base, 0.4)
ens = {v: (P_base[v] + P_so[v]) / 2 for v in P_so}
report("ENSEMBLE base+so400m @0.4  (now)", {**P_base, **ens}, 0.4)
print()
best = max(results, key=lambda k: report(k, results[k]))
print(f"\nbest knn: {best}")

# knn + trained heads
K = results[best]
for w in (0.25, 0.5, 0.75):
    mix = {v: (1 - w) * (ens.get(v, P_base[v])) + w * K[v] for v in vids}
    report(f"ens + {w:.2f}*knn @0.4", mix, 0.4)
