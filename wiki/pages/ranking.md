# Ranking — why Levels 2 and 3 are stuck, and what does not fix it

## The problem is ordering, not proposals

`scripts/grid_strategy.py` builds candidate windows on a time lattice: starts on
a 2.5 second grid, durations bracketing the real event-length distribution. That
lattice **covers 100% of the public L2 and L3 ground-truth events at IoU ≥ 0.5**.
Proposal generation is solved. Everything that is left is the score used to order
those windows.

That matters because of the shape of the marks formula. A video scores
`0.2*alert + 0.4*F1(matched) + 0.4*mean_IoU(matched)`, and with `k` candidates
and `m` matches the F1 term is about `2m/k`. Spraying every window guarantees the
hit and the timing term but drives F1 to zero, so the score tends to
`0.2 + 0.4*IoU ≈ 0.6`. Ranking well enough to cut `k` to a handful is the only
way to buy the 0.4 back, and it is worth roughly sixteen marks.

## The head cannot rank at all

Recall of the true grid windows using the temporal head's mean anomaly score:

| | @4 | @8 | @16 | @32 | @64 | @128 | @256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D2 | 28% | 39% | 39% | 44% | 61% | 67% | 89% |
| D3 | 12% | 12% | 12% | 12% | 12% | 38% | 50% |

The reason is visible in the curve itself. On **E023, E026, E028** — and on
public **T027 and T032** — the head's anomaly output is saturated at exactly
`1.0000` with a standard deviation of `0.0000`. It is not weakly informative on
those videos; it reports every instant as anomalous, so the ordering of windows
within them is arbitrary. T031, T034 and E027 are 82-90% above 0.99, which is
nearly as bad.

This is a training-distribution failure, not a tuning problem. The head learned
on synthetic concatenations of ~5 second clips in which anomalous frames were
dense, and on genuine 240-629 second footage it has collapsed to always-on.

## Ten replacement scores, benchmarked

From `scripts/window_rank.py`, ranking **(window, class) pairs** rather than
windows, because a perfectly placed window with the wrong label is worth nothing.

| ranker | D2 recall@8 / @128 | D2 best x | D3 recall@8 / @128 | D3 best x |
|---|---|---:|---|---:|
| head mean (baseline) | 22% / 39% | 0.600 | 0% / 0% | 0.567 |
| clip classifier `P(class│window)` | 33% / 33% | 0.600 | 0% / 0% | 0.567 |
| background deviation | 11% / 28% | 0.598 | 0% / 0% | 0.567 |
| background deviation, contrast | 17% / 28% | 0.598 | 0% / 0% | 0.567 |
| **clip probability, contrast** | **33% / 50%** | **0.649** | 0% / 0% | 0.565 |
| class prototype cosine | 6% / 28% | 0.598 | 0% / 0% | 0.567 |
| text tower, max in window | 0% / 11% | 0.598 | 12% / 12% | 0.567 |
| text tower, contrast | 11% / 50% | 0.598 | 0% / 0% | 0.557 |
| clip contrast, max-pooled | 6% / 39% | 0.598 | 12% / 12% | 0.567 |
| fuse clip + background contrast | 22% / 33% | 0.598 | 0% / 0% | 0.567 |
| fuse clip + clip contrast | 28% / 44% | 0.598 | 0% / 0% | 0.567 |
| fuse clip contrast + text | 28% / 56% | 0.638 | 0% / 12% | 0.565 |

**On D3 nothing works.** Every one of the twelve scores is at or below 12% recall
at k=128 and none moves `x` off the spray ceiling. Background deviation, which
should be the natural fit for static CCTV, is the worst of them. This is the
clearest evidence we have that the frozen SigLIP representation does not separate
these events from their own background at all, and no amount of decoding on top
of it will.

**On D2 one score wins**: the clip classifier's probability contrasted against its
own local neighbourhood, emitting only the top four pairs — 0.602 → 0.649, which
projects D2 from 24.6 to 25.8 of 35. The K sweep falls smoothly (0.649, 0.614,
0.590, 0.561 from K=4), so it is a plateau rather than a lucky cell, but the gain
is carried by T028 scoring a perfect 1.000 while T025 scores 0.200 at every K
because every signal we have calls its `traffic_accident` events
`wrong_way_driving`. Four videos is not enough to call this settled, which is why
it ships as a second file rather than a replacement.

Suppression matters when committing to few windows. Without it the top four pairs
on E021 were 220-240, 220-230, 220-235 and 218-238 — four claims on one moment,
where the rules allow only the best-overlapping fragment to match and count the
rest against you. Suppressing at IoU 0.3 changes nothing on the public set, where
the top four were already spread, and is what makes the private set behave.

## Composition search

`scripts/periodic.py` searches whole hypotheses — `n` events of length `d`
starting at `a` and repeating every `b` — instead of scoring windows one at a
time, on the theory that averaging a noisy curve over four to six windows is
much steadier than reading it at one. See [[fingerprints]] for why the collection
is composed at all.

It works perfectly where the composition is regular. On T025 the top-ranked
hypothesis is `n=6, a=20, b=40, d=20`, which *is* the ground truth, at rank 0;
on T028 it returns `n=4, a=30, b=60, d=5`, also exactly right, at rank 0. On
T026 and T027, which are not periodic, it fails. **E021's top hypothesis is
`n=6, a=20, b=40, d=20` with contrast +0.898 — the identical pattern to T025.**

Two constraints stop it degenerating: events may not overlap (`b ≥ d`), and the
progression must span at least half the video. Without the first, the best
hypothesis on E022 was eight windows overlapping at a five-second period, which
is one spike wearing a progression's clothes.

It is not yet a reliable win — two of four public videos, and the class is wrong
on T025 — so the lattice still ships. It is the most promising lead left, and it
becomes a large win the moment the class prediction on that collection improves,
because a correct hypothesis with a correct class scores 1.000 against the
lattice's 0.60.

## What would actually break the ceiling

Only a representation that localises. The blocked LoRA fine-tune on real long
footage (see [[milestones]] M4) is the one lever that addresses the cause rather
than decoding harder around it.
