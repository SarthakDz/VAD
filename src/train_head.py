"""Train the temporal head on synthesised long sequences.

Supervision is dense: the audit confirmed all 2200 anomaly rows carry
timestamps, so there is no need for the MIL top-k pooling the PRD planned for
video-level-only labels. Every timestep has a target.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset_head import ClipBank, SyntheticSequences
from .head import TemporalHead
from .labels import CLASSES


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_bank(bank: ClipBank, val_frac: float, seed: int):
    """Hold out whole clips. Several classes look like single-source benchmark
    rips (waterlogging is 95 clips all under 2.3 MB, loitering 300 in a narrow
    9-15 MB band), so a random split leaks source characteristics. Stratifying
    by class at least keeps every class represented on both sides."""
    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for a in bank.anomaly:
        by_class.setdefault(a["class_idx"], []).append(a)
    train, val = [], []
    for items in by_class.values():
        rng.shuffle(items)
        k = max(1, int(len(items) * val_frac)) if len(items) > 1 else 0
        val += items[:k]
        train += items[k:]
    return train, val


def make_loss(pos_weight: float, device: str):
    bce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(pos_weight, device=device)
    )
    ce = nn.CrossEntropyLoss(reduction="none")

    def fn(a_logit, c_logit, y_a, y_c, w):
        la = (bce(a_logit, y_a) * w).mean()
        lc = (ce(c_logit.reshape(-1, c_logit.shape[-1]), y_c.reshape(-1))
              * w.reshape(-1)).mean()
        return la + lc, la.item(), lc.item()

    return fn


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn_ = 0
    cls_ok = cls_n = 0
    for x, ya, yc, _ in loader:
        x, ya, yc = x.to(device), ya.to(device), yc.to(device)
        a, c = model(x)
        pred = (torch.sigmoid(a) > 0.5).float()
        tp += ((pred == 1) & (ya == 1)).sum().item()
        fp += ((pred == 1) & (ya == 0)).sum().item()
        fn_ += ((pred == 0) & (ya == 1)).sum().item()
        m = ya == 1
        if m.any():
            cls_ok += (c.argmax(-1)[m] == yc[m]).sum().item()
            cls_n += int(m.sum().item())
    f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn_)
    return {"frame_f1": f1, "frame_cls_acc": cls_ok / max(1, cls_n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Train and Test")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="outputs/head.pt")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--p-normal", type=float, default=0.5)
    ap.add_argument("--localised-weight", type=float, default=3.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--causal", action="store_true", help="unidirectional GRU")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    seed_all(a.seed)
    bank = ClipBank(a.root, a.cache)
    print(bank)
    tr_items, va_items = split_bank(bank, a.val_frac, a.seed)
    print(f"clips: {len(tr_items)} train / {len(va_items)} val")

    tr_bank, va_bank = ClipBank.__new__(ClipBank), ClipBank.__new__(ClipBank)
    for dst, items in ((tr_bank, tr_items), (va_bank, va_items)):
        dst.__dict__.update(bank.__dict__)
        dst.anomaly = items

    tr = SyntheticSequences(tr_bank, a.window, a.steps_per_epoch * a.batch_size,
                            a.p_normal, a.localised_weight, a.seed)
    va = SyntheticSequences(va_bank, a.window, 400, a.p_normal, 1.0, a.seed + 1)
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=False, num_workers=0)
    vl = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=0)

    # Anomaly frames are the minority; measure the ratio rather than guess it.
    frac = float(np.mean([tr[i][1].mean().item() for i in range(64)]))
    pos_w = max(1.0, (1 - frac) / max(frac, 1e-3))
    print(f"anomaly frame fraction {frac:.3f}  ->  pos_weight {pos_w:.2f}")

    model = TemporalHead(tr.dim, a.hidden, a.layers, a.dropout,
                         bidirectional=not a.causal).to(a.device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(tl), pct_start=0.2)
    loss_fn = make_loss(pos_w, a.device)

    print(f"head {n_par/1e6:.2f}M params, "
          f"{'causal' if a.causal else 'bidirectional'}, on {a.device}")

    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = na = nc = 0.0
        for x, ya, yc, w in tl:
            x, ya, yc, w = (t.to(a.device) for t in (x, ya, yc, w))
            al, cl = model(x)
            loss, la, lc = loss_fn(al, cl, ya, yc, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item(); na += la; nc += lc
        m = evaluate(model, vl, a.device)
        n = len(tl)
        flag = ""
        if m["frame_f1"] > best:
            best = m["frame_f1"]
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "config": {"input_dim": tr.dim, "hidden": a.hidden,
                                   "layers": a.layers, "dropout": a.dropout,
                                   "bidirectional": not a.causal},
                        "classes": CLASSES, "val": m}, a.out)
            flag = "  *saved"
        print(f"ep {ep:2d}  loss {tot/n:.4f} (a {na/n:.4f} c {nc/n:.4f})  "
              f"val frame_f1 {m['frame_f1']:.4f}  cls_acc {m['frame_cls_acc']:.4f}{flag}")

    print(f"\nbest val frame_f1 {best:.4f} in {(time.time()-t0)/60:.1f} min -> {a.out}")
    json.dump({"best_frame_f1": best, "params": n_par},
              open(Path(a.out).with_suffix(".json"), "w"))


if __name__ == "__main__":
    main()
