"""Train the clip-level classifier and report held-out accuracy.

Held-out accuracy here is the direct predictor of Difficulty 1, which is a pure
clip-classification task worth 25 marks that the temporal head currently
converts at 13.2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .clip_classifier import ClipHead, build_dataset
from .labels import CLASSES


def stratified_split(y: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_frac)) if len(idx) > 1 else 0
        va += list(idx[:k])
        tr += list(idx[k:])
    return np.array(tr), np.array(va)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Train and Test")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="outputs/clip.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corrections", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    t0 = time.time()
    X, y, _ = build_dataset(a.root, a.cache, apply_corrections=a.corrections)
    print(f"{len(X)} clips, dim {X.shape[1]}, built in {time.time()-t0:.1f}s")
    counts = np.bincount(y, minlength=len(CLASSES))
    print("  " + "  ".join(f"{CLASSES[i][:12]}:{counts[i]}" for i in range(len(CLASSES)) if counts[i]))

    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    tr, va = stratified_split(y, a.val_frac, a.seed)
    Xtr = torch.from_numpy(X[tr]).float().to(a.device)
    ytr = torch.from_numpy(y[tr]).long().to(a.device)
    Xva = torch.from_numpy(X[va]).float().to(a.device)
    yva = torch.from_numpy(y[va]).long().to(a.device)
    print(f"  train {len(tr)}  val {len(va)}")

    model = ClipHead(X.shape[1], a.hidden, len(CLASSES), a.dropout).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # Inverse-frequency weights: normal is 973 clips against fire's 77.
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5,
                     dtype=torch.float32, device=a.device)
    w = w / w.mean()
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    best, best_state = -1.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=a.device)
        tot = 0.0
        for i in range(0, len(perm), a.batch_size):
            b = perm[i:i + a.batch_size]
            loss = lossf(model(Xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        model.eval()
        with torch.inference_mode():
            pv = model(Xva).argmax(-1)
            acc = (pv == yva).float().mean().item()
            anom = yva != 0
            acc_a = (pv[anom] == yva[anom]).float().mean().item() if anom.any() else 0.0
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  loss {tot/max(1,len(perm)//a.batch_size):.4f}  "
                  f"val_acc {acc:.4f}  anomaly_only {acc_a:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pv = model(Xva).argmax(-1).cpu().numpy()
    yv = yva.cpu().numpy()
    print(f"\nbest val accuracy {best:.4f}")
    print(f"{'class':34s}{'acc':>7}{'n':>5}")
    for c in range(len(CLASSES)):
        m = yv == c
        if m.sum():
            print(f"  {CLASSES[c]:32s}{(pv[m]==c).mean():7.3f}{m.sum():5d}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd,
                "config": {"dim": X.shape[1], "hidden": a.hidden,
                           "n_classes": len(CLASSES), "dropout": a.dropout},
                "val_acc": best, "classes": CLASSES}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
