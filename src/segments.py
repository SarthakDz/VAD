"""Per-timestep score curve -> event segments.

This module decides the score at Levels 2 and 3, more than detection accuracy
does. The sanity check in scripts/sanity_score.py made that concrete: a
*perfect* set of events, correct classes and all, shattered into five slices
each, collapses L2 from 1.000 to 0.467 and L3 from 1.000 to 0.200. Only the
best-overlapping fragment can match and the rest count as false positives.

So the bias here is deliberately toward few, long, merged segments:
  * hysteresis -- a high `enter` threshold to start, a lower `exit` to
    continue, so one event does not flicker into forty
  * merge anything separated by less than `merge_gap_sec`
  * drop anything shorter than `min_event_sec`
  * one class per segment, taken as the score-weighted majority
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import CLASSES, NORMAL


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    class_name: str
    score: float
    lo: int
    hi: int


def hysteresis(scores: np.ndarray, enter: float, exit_: float) -> list[tuple[int, int]]:
    """Half-open [lo, hi) index spans where the curve is 'on'."""
    spans, lo, on = [], 0, False
    for i, s in enumerate(scores):
        if not on and s >= enter:
            on, lo = True, i
        elif on and s < exit_:
            spans.append((lo, i))
            on = False
    if on:
        spans.append((lo, len(scores)))
    return spans


def merge(spans: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not spans:
        return []
    out = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo - out[-1][1] <= gap:
            out[-1][1] = hi
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def extract(
    anomaly: np.ndarray,
    class_prob: np.ndarray,
    timestamps: np.ndarray,
    enter: float = 0.70,
    exit_: float = 0.45,
    merge_gap_sec: float = 5.0,
    min_event_sec: float = 2.0,
    split_on_class_change: bool = True,
) -> list[Segment]:
    """Score curves -> segments. `timestamps[i]` is the wall-clock second of
    timestep i, so every boundary comes back in real seconds."""
    if len(anomaly) == 0:
        return []

    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.5
    step = max(step, 1e-3)
    gap_steps = max(1, int(round(merge_gap_sec / step)))

    spans = merge(hysteresis(anomaly, enter, exit_), gap_steps)

    out: list[Segment] = []
    for lo, hi in spans:
        for a, b in (_split_by_class(class_prob, lo, hi) if split_on_class_change
                     else [(lo, hi)]):
            cls = _dominant_class(class_prob, anomaly, a, b)
            if cls == NORMAL:
                continue
            s, e = float(timestamps[a]), float(timestamps[min(b, len(timestamps) - 1)])
            if e - s < min_event_sec:
                continue
            out.append(Segment(s, e, cls, float(anomaly[a:b].mean()), a, b))
    return out


def _dominant_class(class_prob: np.ndarray, anomaly: np.ndarray, lo: int, hi: int) -> str:
    """Score-weighted vote, ignoring the `normal` column -- inside a span the
    question is which anomaly it is, not whether it is one."""
    w = anomaly[lo:hi][:, None]
    mass = (class_prob[lo:hi] * w).sum(axis=0)
    mass[CLASSES.index(NORMAL)] = -np.inf
    return CLASSES[int(np.argmax(mass))]


def _split_by_class(class_prob: np.ndarray, lo: int, hi: int,
                    min_run: int = 4) -> list[tuple[int, int]]:
    """Cut a span where the winning class changes and holds.

    T026 carries four different classes in one video, so a single continuous
    high-score region can legitimately be several events. `min_run` stops
    per-frame jitter from fragmenting a span -- the very failure mode trap 4
    punishes.
    """
    normal_idx = CLASSES.index(NORMAL)
    p = class_prob[lo:hi].copy()
    p[:, normal_idx] = -np.inf
    winners = p.argmax(axis=1)

    cuts, run_start = [], 0
    for i in range(1, len(winners)):
        if winners[i] != winners[run_start]:
            if i - run_start >= min_run:
                cuts.append((run_start, i))
                run_start = i
            else:
                winners[run_start:i] = winners[i]  # absorb a short blip
                run_start = run_start
    cuts.append((run_start, len(winners)))
    return [(lo + a, lo + b) for a, b in cuts if b > a]


def to_events(segments: list[Segment], level: int, explanations: dict | None = None):
    """Segments -> arena event dicts. Level 1 must carry null timestamps."""
    from .submit import Event

    if level == 1:
        if not segments:
            return []
        best = max(segments, key=lambda s: s.score * (s.end_sec - s.start_sec))
        ex = (explanations or {}).get(best.class_name)
        return [Event(best.class_name, None, None, ex)]

    return [Event(s.class_name, round(s.start_sec, 2), round(s.end_sec, 2),
                  (explanations or {}).get(s.class_name)) for s in segments]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 200
    a = np.full(T, 0.1)
    a[40:70] = 0.9
    a[72:75] = 0.9          # blip that must merge into the block above
    a[150:160] = 0.9
    a += rng.normal(0, 0.02, T)
    c = np.full((T, 12), 0.01)
    c[:, 0] = 0.9
    c[40:75, 0] = 0.1; c[40:75, 1] = 0.9
    c[150:160, 0] = 0.1; c[150:160, 8] = 0.9
    ts = np.arange(T) * 0.5
    for s in extract(a, c, ts):
        print(f"  {s.start_sec:6.1f}s - {s.end_sec:6.1f}s  {s.class_name:20s} {s.score:.2f}")
