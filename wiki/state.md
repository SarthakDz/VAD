# State — where the work stands

**Last updated:** 2026-09-05 18:30

## Position

Live board is the private **Evaluation pack**. Standing score **47.6** (v8).

```
v8         D1 11.8  D2 16.1  D3 19.7  = 47.6   <- counts now
v6_lean    D1 13.4  D2 16.4  D3 16.3  = 46.1
v7_lean    D1 11.8  D2 16.4  D3 16.3  = 44.5
v4         D1 13.4  D2 16.1  D3 16.7  = 46.2
```

Best per-difficulty across eleven uploads is 13.4 + 16.4 + 19.7 = **49.5**, so a
hybrid alone was worth +1.9. The full Level-3 lattice is what took D3 from 16.3
to 19.7; the lean version is not worth using.

## Next action — upload all three, in any order

```
outputs/submission_v9a.json       silences E021 and E024 on D2
outputs/submission_v9b.json       silences E022 and E024 on D2
outputs/submission_v9probe.json   Level 3 silent -- a measurement, not an entry
```

Exactly one of v9a and v9b should jump about 8 marks on D2 and the other should
drop about 4. Best run stands, so trying both is strictly better than choosing.
The probe returns D3 as (normal L3 videos)/4 * 40, so 0.0, 10.0, 20.0 or 30.0.
If it is not 0.0 there is a normal Level-3 video worth ten marks to silence.

## The two corrections v9 is built on

**Level 1 is not an F1.** The leaderboard prints `found x/17`, so 17 of the 20
Level-1 videos are anomalous. The real rule is the format PDF's own and it
reproduces four uploads to within 0.02 marks:

```
D1 = 25 * [ 0.5*binary_accuracy(/20) + 0.5*class_accuracy(/17 anomalous) ]
```

Half the marks are binary accuracy and 85% of the videos carry an event, so
**there is no precision penalty at Level 1** and every confidence threshold we
ever used was throwing marks away. v9 claims on all eighteen videos that are not
E002 or E004, the two normals the upload deltas identify. This also explains why
the rebuilt classifier in [[d1]] lost marks on the private set while winning on
public: it optimises F1, which is the wrong objective.

**Two of the four Level-2 videos are normal.** A rival predicted nothing at all
on D2 (`found 0/12, FA 0`) and scored exactly 17.5/35 = 0.500, which is two
correct silences out of four. E024 is one; `submission_asym` pins the other to
E021 or E022 and proves E023 anomalous. The alert weight is 0.30. Every upload we
have made put events on both E021 and E022, so one of them has scored zero every
time.

## What is still capped

Levels 2 and 3 candidate windows sit on the 2.5 s lattice the ground truth is
composed on (see [[fingerprints]]), which covers 100% of the public truths at
IoU >= 0.5. Ranking is what fails: the head's anomaly curve is saturated at
exactly 1.0000 with zero variance on E023, E026 and E028, and twelve replacement
scores all reach 0% recall at k=128 on D3 ([[ranking]]). With k large the score
tends to 0.2 + 0.4*IoU, so the sprayed videos are pinned near 0.6.

Real headroom now sits in the **structural facts the leaderboard keeps leaking**,
not in the models: normal videos are worth a full video each, and there are only
6 ground-truth events at Level 3 and 12 at Level 2.

## Blocked on the user

- All arena uploads; no session can reach the site
- Kaggle phone verification for M4, the LoRA fine-tune
- 2-slide PPT and architecture write-up sign-off

## Settled questions — do not re-ask

- Level weighting D1 25, D2 35, D3 40; best **run** stands, so uploads are free.
- 17 of 20 L1 videos are anomalous; E002 and E004 are two of the three normals.
- Two of four L2 videos are normal: E024, and one of E021/E022. E023 is anomalous.
- D2 alert weight is 0.30. `src/calibrated.py`'s F1 model for D1 is **wrong** for
  this pack — it fitted the practice pack only.
- Ground-truth event counts: 17 at L1, 12 at L2, 6 at L3, 35 in total.
- Eval videos are not duplicates of anything we hold; only E027 has audio.
