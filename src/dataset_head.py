"""Training sequences for the temporal head, synthesised from short clips.

The central problem this solves, from the data audit:

    train  3173 clips, ONE event each, median 5.3s, max 30s
    test   Levels 2-3 are 240-629s, multi-event AND multi-class
           (T026 carries four different classes; T025 six separate accidents)

Nothing in the training set teaches a temporal model about transitions, about
two classes in one sequence, or about long normal stretches between events --
which is exactly what Levels 2 and 3 score. So we build those sequences
ourselves by concatenating clip embeddings, carrying the offset arithmetic
through to exact interval labels. Concatenation happens in embedding space,
so it costs microseconds and no video is ever re-decoded.

Second audit finding this handles: 1551 of 2200 anomaly events start at 0.000
and run the whole clip, so their interval carries no localisation signal. The
~649 clips with start > 0 are the only real boundary supervision, and they get
`localised_weight` applied to their timesteps.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import CLASS_TO_IDX, NORMAL


class ClipBank:
    """Embeddings plus frame-level labels for every cached training clip."""

    def __init__(self, root: str | Path, cache: str | Path, min_frames: int = 4):
        from .io_dataset import load_train

        self.emb_dir = Path(cache) / "emb"
        self.meta_dir = Path(cache) / "meta"
        df = load_train(root)

        self.anomaly: list[dict] = []
        self.normal: list[dict] = []
        self.long_normal: list[dict] = []
        missing = 0

        for r in df.itertuples():
            p = self.emb_dir / f"{r.video_id}.npy"
            mp = self.meta_dir / f"{r.video_id}.json"
            if not p.exists() or not mp.exists():
                missing += 1
                continue
            meta = json.loads(mp.read_text())
            n = int(meta["sampled_frames"])
            if n < min_frames:
                continue
            dur = float(meta["duration_sec"]) or 1e-6

            item = {
                "video_id": r.video_id,
                "path": p,
                "n": n,
                "duration": dur,
                "class_name": r.class_name,
                "class_idx": CLASS_TO_IDX[r.class_name],
            }
            if r.class_name == NORMAL:
                (self.long_normal if n >= 240 else self.normal).append(item)
            else:
                s = 0.0 if np.isnan(r.start_time_sec) else float(r.start_time_sec)
                e = dur if np.isnan(r.end_time_sec) else float(r.end_time_sec)
                lo = max(0, int(round(s / dur * n)))
                hi = min(n, max(lo + 1, int(round(e / dur * n))))
                item.update(lo=lo, hi=hi, localised=s > 0.0 or e < dur * 0.99)
                self.anomaly.append(item)

        self.missing = missing

    def load(self, item: dict) -> np.ndarray:
        return np.load(item["path"]).astype(np.float32)

    def __repr__(self) -> str:
        loc = sum(1 for a in self.anomaly if a["localised"])
        return (f"ClipBank(anomaly={len(self.anomaly)} [{loc} localised], "
                f"normal={len(self.normal)}, long_normal={len(self.long_normal)}, "
                f"uncached={self.missing})")


class SyntheticSequences(Dataset):
    """Random long multi-event sequences assembled from the clip bank."""

    def __init__(
        self,
        bank: ClipBank,
        window: int = 256,
        length: int = 4000,
        p_normal: float = 0.5,
        localised_weight: float = 3.0,
        seed: int = 1337,
    ):
        self.bank = bank
        self.window = window
        self.length = length
        self.p_normal = p_normal
        self.localised_weight = localised_weight
        self.seed = seed
        if not bank.anomaly:
            raise RuntimeError("clip bank has no cached anomaly clips -- run src.encode first")
        self.dim = bank.load(bank.anomaly[0]).shape[1]

        # Sample anomaly clips inversely to class frequency so traffic_accident
        # (565 clips) does not drown out fire (77).
        counts: dict[int, int] = {}
        for a in bank.anomaly:
            counts[a["class_idx"]] = counts.get(a["class_idx"], 0) + 1
        self.weights = [1.0 / counts[a["class_idx"]] for a in bank.anomaly]

    def __len__(self) -> int:
        return self.length

    def _normal_chunk(self, rng: random.Random, need: int):
        """A stretch of normal footage, preferring real long normal video."""
        pool = self.bank.long_normal if (self.bank.long_normal and rng.random() < 0.5) \
            else (self.bank.normal or self.bank.long_normal)
        if not pool:
            return None
        item = rng.choice(pool)
        emb = self.bank.load(item)
        if len(emb) > need:
            start = rng.randrange(0, len(emb) - need)
            emb = emb[start:start + need]
        return emb

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed * 1_000_003 + idx)
        W, dim = self.window, self.dim

        x = np.zeros((W, dim), dtype=np.float32)
        y_anom = np.zeros(W, dtype=np.float32)
        y_cls = np.zeros(W, dtype=np.int64)      # 0 == normal
        w = np.ones(W, dtype=np.float32)

        t = 0
        guard = 0
        while t < W and guard < 200:
            guard += 1
            need = W - t

            if rng.random() < self.p_normal:
                emb = self._normal_chunk(rng, need)
                if emb is None:
                    break
                n = min(len(emb), need)
                x[t:t + n] = emb[:n]
                t += n
                continue

            item = rng.choices(self.bank.anomaly, weights=self.weights, k=1)[0]
            emb = self.bank.load(item)
            n = min(len(emb), need)
            x[t:t + n] = emb[:n]

            lo, hi = item["lo"], min(item["hi"], n)
            if hi > lo:
                y_anom[t + lo:t + hi] = 1.0
                y_cls[t + lo:t + hi] = item["class_idx"]
                if item["localised"]:
                    w[t + lo:t + hi] = self.localised_weight
            t += n

        if t < W:  # bank exhausted early -- pad with the last normal stretch
            emb = self._normal_chunk(rng, W - t)
            if emb is not None:
                n = min(len(emb), W - t)
                x[t:t + n] = emb[:n]

        return (torch.from_numpy(x), torch.from_numpy(y_anom),
                torch.from_numpy(y_cls), torch.from_numpy(w))


class RealClips(Dataset):
    """The raw clips, unconcatenated -- a held-out sanity set."""

    def __init__(self, bank: ClipBank, items: list[dict], max_len: int = 256):
        self.bank, self.items, self.max_len = bank, items, max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        emb = self.bank.load(item)[: self.max_len]
        n = len(emb)
        y_anom = np.zeros(n, dtype=np.float32)
        y_cls = np.zeros(n, dtype=np.int64)
        if "lo" in item:
            lo, hi = item["lo"], min(item["hi"], n)
            if hi > lo:
                y_anom[lo:hi] = 1.0
                y_cls[lo:hi] = item["class_idx"]
        return (torch.from_numpy(emb.astype(np.float32)), torch.from_numpy(y_anom),
                torch.from_numpy(y_cls), torch.ones(n))


if __name__ == "__main__":
    import sys

    bank = ClipBank(sys.argv[1] if len(sys.argv) > 1 else "../Train and Test", "cache")
    print(bank)
    ds = SyntheticSequences(bank, window=256, length=4)
    for i in range(2):
        x, ya, yc, w = ds[i]
        present = sorted(set(yc.tolist()) - {0})
        print(f"  seq {i}: x{tuple(x.shape)} anomaly_frac {ya.mean():.2f} "
              f"classes {present} weight_max {w.max():.1f}")
