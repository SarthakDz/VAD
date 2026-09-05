"""A clip-level classifier, and a sliding window built from it.

Why this exists alongside the GRU head.

The temporal head is trained on synthesised 256-timestep concatenations (~128 s)
because the test set's Levels 2 and 3 are long. But **every training clip is
about 5 seconds long**, so those concatenations are a construct: the model never
sees a real 128-second scene, only stitched short ones.

This module inverts the trade. Train a classifier on exactly what the data
actually is — one short clip, one label — and then apply it to long videos by
sliding a window of the same length. The train and inference distributions
match by construction, which the GRU's never did.

  D1: one window over the whole clip -> a single label. This is the pure
      classification task the difficulty actually asks for, and it is worth
      25 marks that the head currently converts at 13.2.
  D2/D3: slide, classify each window, then merge runs of the same class into
      events. Boundaries come from where the classification changes.

Features are mean+max pooling over the frozen SigLIP embeddings of a window.
Max-pooling matters: an accident is a one-second event inside a five-second
clip, and mean-pooling alone washes it out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .labels import CLASS_TO_IDX, CLASSES, NORMAL


def pool(emb: np.ndarray) -> np.ndarray:
    """(T, D) -> (2D,) mean and max concatenated."""
    if emb.ndim == 1:
        emb = emb[None, :]
    return np.concatenate([emb.mean(axis=0), emb.max(axis=0)])


class ClipHead(nn.Module):
    def __init__(self, dim: int = 1536, hidden: int = 512,
                 n_classes: int = len(CLASSES), dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_dataset(root: str | Path, cache: str | Path, apply_corrections: bool = False):
    """One feature vector per training clip, pooled over its event interval.

    Normal clips pool over the whole clip. Anomaly clips pool over
    [start_time_sec, end_time_sec] only -- that interval is the label's actual
    support, and 649 of them are genuinely narrower than the clip.
    """
    from .io_dataset import load_train

    emb_dir, meta_dir = Path(cache) / "emb", Path(cache) / "meta"
    df = load_train(root, apply_corrections=apply_corrections)
    X, y, groups = [], [], []
    for r in df.itertuples():
        p, mp = emb_dir / f"{r.video_id}.npy", meta_dir / f"{r.video_id}.json"
        if not p.exists() or not mp.exists():
            continue
        emb = np.load(p).astype(np.float32)
        n = len(emb)
        if n == 0:
            continue
        meta = json.loads(mp.read_text())
        dur = float(meta["duration_sec"]) or 1e-6
        lo, hi = 0, n
        if r.class_name != NORMAL and not np.isnan(r.start_time_sec):
            lo = int(round(float(r.start_time_sec) / dur * n))
            hi = int(round(float(r.end_time_sec) / dur * n))
            # A short event in a short clip can round to an empty or inverted
            # span; fall back to the whole clip rather than dropping the sample.
            lo = min(max(0, lo), n - 1)
            hi = min(max(lo + 1, hi), n)
            if hi <= lo:
                lo, hi = 0, n
        X.append(pool(emb[lo:hi]))
        y.append(CLASS_TO_IDX[r.class_name])
        groups.append(r.source_class)
    return np.stack(X), np.array(y), np.array(groups)


@torch.inference_mode()
def classify_windows(emb: np.ndarray, model: ClipHead, timestamps: np.ndarray,
                     win_sec: float = 6.0, stride_sec: float = 2.0,
                     device: str = "cuda"):
    """Slide a window over one video. Returns (centres, probs (W, 12))."""
    if len(emb) == 0:
        return np.zeros(0), np.zeros((0, len(CLASSES)))
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.5
    step = max(step, 1e-3)
    w = max(1, int(round(win_sec / step)))
    s = max(1, int(round(stride_sec / step)))

    feats, centres = [], []
    for lo in range(0, max(1, len(emb) - w + 1), s):
        hi = min(len(emb), lo + w)
        feats.append(pool(emb[lo:hi]))
        centres.append(float(timestamps[min(len(timestamps) - 1, (lo + hi) // 2)]))
    if not feats:
        feats, centres = [pool(emb)], [float(timestamps[len(timestamps) // 2])]

    x = torch.from_numpy(np.stack(feats)).float().to(device)
    probs = torch.softmax(model(x), dim=-1).cpu().numpy()
    return np.array(centres), probs


def windows_to_segments(centres: np.ndarray, probs: np.ndarray, timestamps: np.ndarray,
                        threshold: float = 0.5, min_run: int = 2,
                        merge_gap_sec: float = 10.0, min_event_sec: float = 2.0):
    """Runs of consecutive windows sharing a confident anomaly class -> segments."""
    from .segments import Segment

    if len(centres) == 0:
        return []
    normal_idx = CLASS_TO_IDX[NORMAL]
    best = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    active = (best != normal_idx) & (conf >= threshold)

    runs, i = [], 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(active) and active[j + 1] and best[j + 1] == best[i]:
            j += 1
        runs.append((i, j, int(best[i])))
        i = j + 1

    merged = []
    for lo, hi, c in runs:
        if merged and merged[-1][2] == c and centres[lo] - centres[merged[-1][1]] <= merge_gap_sec:
            merged[-1] = (merged[-1][0], hi, c)
        else:
            merged.append((lo, hi, c))

    out = []
    end_of_video = float(timestamps[-1]) if len(timestamps) else 0.0
    for lo, hi, c in merged:
        if (hi - lo + 1) < min_run:
            continue
        s = max(0.0, float(centres[lo]))
        e = min(end_of_video, float(centres[hi]))
        if e - s < min_event_sec:
            continue
        out.append(Segment(s, e, CLASSES[c], float(conf[lo:hi + 1].mean()), lo, hi + 1))
    return out
