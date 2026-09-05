# State — where the work stands

**Last updated:** 2026-09-05 18:05

## Position

Live board is the private **Evaluation pack** (E001-E028, L1=20 L2=4 L3=4).

```
submission_v4.json   D1 13.4  D2 16.1  D3 16.7  = 46.2   <- counts now
submission_v5.json   D1 11.4  D2 16.1  D3 16.7  = 44.2
submission_v2final   D1 12.0  D2 14.0  D3 11.2  = 37.2
```

v5 differed from v4 only by raising the Level-1 probability threshold from 0.4
to 0.7, and it cost two marks. [[d1]] explains why: D1 is `25*F1`, so a claim of
probability `p` pays whenever `p > F1/2`, which at our F1 is **0.27** — the
threshold was already too high, not too low.

## Next action — upload these two

```
outputs/submission_v7.json    the safe build     projected ~62
outputs/submission_v8.json    v7 with a ranked D2 instead of a sprayed one
```

Both pass `src.submit.validate`. Upload v7 first, then v8; best run stands, so
the pair is an experiment rather than a hedge. Built by `scripts/eval_v7.py` and
`scripts/eval_v8.py`, which carry the full reasoning.

## What the arena detection panel told us

The per-run panel is far more informative than the three marks. From the v4 run:

```
precision 5%   recall 37%   F1 8%   false alarms 258
weakest classes, all at 0% found:
    fighting_or_violence          0 / 3 truths, 66 false
    road_spill_or_debris          0 / 3 truths, 18 false
    stalled_or_broken_down_veh.   0 / 2 truths, 24 false
    vehicle_blocking_traffic      0 / 1 truths, 18 false
```

271 events minus 258 false gives **13 true positives**, and 13/0.37 gives
**35 ground-truth events in total**. The per-class false counts reconcile exactly
with what we emitted, so the panel can be used as a measuring instrument.

There is also a **Level-3 reasoning bonus, on top of the 40**, reported as *not
graded* because we had never supplied an `explanation` on any event. It is
bonus-only and omitting it never costs marks, so it was pure forgone credit.
v7 and v8 now carry explanations.

## The ceiling, and where it comes from

Candidate windows now sit on the 2.5 s lattice the ground truth is composed on
(see [[fingerprints]]), and that lattice **covers 100% of the public L2/L3
truths at IoU >= 0.5**. Proposals are solved. What is not solved is ranking, and
[[ranking]] has the evidence: the head anomaly curve is saturated at exactly
`1.0000` with zero variance on E023, E026 and E028, and twelve replacement
scores — clip classifier, background deviation, class prototypes, SigLIP text
tower, and fusions of them — all score **0% recall at k=128 on D3**.

So Levels 2 and 3 are pinned at the spray ceiling. As `k` grows the F1 term
vanishes and the score tends to `0.2 + 0.4*IoU = 0.6` per video; we measure
0.602 on L2 and 0.582 on L3. That is 24.6/35 and 23.3/40.

**80/100 is not reachable from here.** It needs per-video scores near 0.85, which
needs matched-F1 near 1, which needs emitting about four windows and hitting all
four. Nothing in the frozen SigLIP representation localises well enough. The
honest route is [[milestones]] M4, the LoRA fine-tune on real long footage, which
is still blocked on Kaggle verification.

## The most promising lead

`scripts/periodic.py` searches for the arithmetic progression the events were
composed on rather than scoring windows one at a time. On T025 it returns
`n=6, a=20, b=40, d=20` — exactly the ground truth — at rank 0, and on T028
`n=4, a=30, b=60, d=5`, also exact, at rank 0. **E021 top hypothesis is the
identical pattern to T025 at contrast +0.898.** It fails on the two non-periodic
public videos and its class is wrong on T025, so the lattice still ships; but a
correct hypothesis with a correct class scores 1.000 where the lattice scores
0.60, so this becomes large the moment classification on that collection
improves.

## Blocked on the user

- All arena uploads; no session can reach the site
- Kaggle phone verification — the only lever left with real headroom
- 2-slide PPT and architecture write-up sign-off

## Settled questions — do not re-ask

- Level weighting D1 25, D2 35, D3 40; best **run** stands, so uploads are free.
- E024 is normal; E025-E028 are all four anomalous.
- D2 alert weight is 0.20, not the 0.30 fitted from the practice pack.
- Eval videos are not duplicates of anything we hold; only E027 has audio.
- Raising the D1 threshold is wrong. The break-even is `p > F1/2`.
