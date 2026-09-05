# State — where the work stands

**Last updated:** 2026-09-05 18:55

## Position

**61.8 / 100 and 2nd place** on the private Evaluation pack, from `v9a`.

```
v9a       D1 14.5  D2 27.6  D3 19.7  = 61.8   <- counts now
v9b       D1 14.5  D2 13.8  D3 19.7  = 48.0
v9probe   D1 14.5  D2 27.6  D3  0.0  = 42.1
v8        D1 11.8  D2 16.1  D3 19.7  = 47.6
```

Both open questions are closed. **E021 is the second normal Level-2 video** —
silencing it took D2 from 16.1 to 27.6, the best Difficulty-2 mark on the board,
and v9b's 13.8 confirms it was not E022. **All four Level-3 videos are
anomalous** — the probe returned exactly 0.0.

## Next action — two things, neither of them modelling

1. **Upload `outputs/submission_v10.json`.** It is v9a with a reasoning string on
   every one of its 19,778 events instead of 66 of them. The REASON column sits
   outside the 100 and the leader takes +4.0 from it while ours reads "-": first
   place is 58.1 + 4.0 against our 61.8 + nothing, so the bonus is the entire
   gap. 3.99 MB against a 5 MB cap.
2. **Fill in the Final Submission.** Our row reads **NONE** where the rest of the
   field reads IN. It wants the repository URL `github.com/SarthakDz/VAD`, the
   architecture write-up (published at
   `https://claude.ai/code/artifact/2bd4ee9f-d91a-445b-96d0-df781d70f79c`, or
   upload `deck/architecture.html`) and optional notes. Required, and separate
   from the score.

## Where the remaining marks are

```
D1  14.5   P 50%  R 53%  found  9/17  FA     9
D2  27.6   P  0%  R 42%  found  5/12  FA  1275
D3  19.7   P  0%  R 83%  found  5/6   FA 18475
```

**Our recall is the highest in the field at both temporal difficulties** and it
converts worst. Nobody else exceeds 2/6 or 3/12, so the lattice genuinely finds
the events; we simply cannot say which of our candidates are the right ones, and
the matched-F1 term therefore collects nothing. A rival turns 2/6 with 2 false
alarms into 22.8 where our 5/6 with 18,475 earns 19.7. About nine marks at Level
3 and four at Level 2 sit behind that single problem, and [[ranking]] records
twelve failed attempts on it.

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
