"""Temporal head over frozen SigLIP embeddings.

Input  (B, T, D)   cached per-frame embeddings, D=768 for siglip-base
Output (B, T, 1)   per-timestep anomaly logit
       (B, T, 12)  per-timestep class logits (index 0 is `normal`)

Bidirectional by default. The arena scores wall-clock processing time, not
causality, and every video is a finished file -- so looking ahead is allowed
and materially improves boundary placement, which is what the 0.5 IoU gate
actually rewards. Set `bidirectional: false` for a causal streaming demo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .labels import CLASSES


class TemporalHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        n_classes: int = len(CLASSES),
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.anomaly = nn.Linear(out_dim, 1)
        self.classes = nn.Linear(out_dim, n_classes)

    def forward(self, x: torch.Tensor):
        h = self.proj(x)
        h, _ = self.rnn(h)
        h = self.norm(h)
        return self.anomaly(h).squeeze(-1), self.classes(h)


@torch.inference_mode()
def predict(model: TemporalHead, emb: torch.Tensor, device: str = "cuda",
            chunk: int = 4096):
    """Score one video. Returns (anomaly_prob (T,), class_prob (T, 12)).

    Chunked so a 20-minute video cannot blow up VRAM. Chunks overlap by
    `pad` timesteps and the overlap is discarded, so the GRU has context on
    both sides of every kept timestep and boundaries do not develop seams.
    """
    model.eval()
    T = emb.shape[0]
    if T <= chunk:
        a, c = model(emb.unsqueeze(0).to(device))
        return torch.sigmoid(a[0]).cpu(), torch.softmax(c[0], dim=-1).cpu()

    pad = 128
    a_out = torch.zeros(T)
    c_out = torch.zeros(T, model.classes.out_features)
    for start in range(0, T, chunk):
        lo = max(0, start - pad)
        hi = min(T, start + chunk + pad)
        a, c = model(emb[lo:hi].unsqueeze(0).to(device))
        a = torch.sigmoid(a[0]).cpu()
        c = torch.softmax(c[0], dim=-1).cpu()
        keep_lo, keep_hi = start - lo, min(chunk, T - start) + (start - lo)
        a_out[start:start + (keep_hi - keep_lo)] = a[keep_lo:keep_hi]
        c_out[start:start + (keep_hi - keep_lo)] = c[keep_lo:keep_hi]
    return a_out, c_out


if __name__ == "__main__":
    m = TemporalHead()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 100, 768)
    a, c = m(x)
    print(f"params {n/1e6:.2f}M   anomaly {tuple(a.shape)}   classes {tuple(c.shape)}")
