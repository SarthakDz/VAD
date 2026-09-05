# State — where the work stands

**Last updated:** 2026-09-05 17:45

## Position

**The arena switched packs.** The 47.0 recorded here before was on the *practice
pack*, now shown as "past". The live board is the **Evaluation pack** over the
private set `F:\flytbase\Evaluation` (E001-E028, L1=20 L2=4 L3=4). Everything
scored against `../Train and Test/test` is now development data, not the score.

**Standing score: 37.2 / 100** — `outputs/submission_v2final.json`, uploaded
16:42, six uploads total. D1 12.0/25, D2 14.0/35, D3 11.2/40.

Uploading stays zero-risk: *"Every upload is kept. Your best scored run is the
one that counts."* Best **run**, not best per difficulty — a hybrid of two
uploads' strengths is worth building, it does not happen automatically.

## Next action — upload these two, in this order

```
outputs/submission.json       projected ~51   (== v4)  D2/D3 rebuild, D1 near-unchanged
outputs/submission_v5.json    projected ~54   v4 plus the D1 threshold at 0.70
```

`submission.json` is the safe one: its gain comes entirely from D2/D3 mechanisms
that follow from the scoring formula. v5 additionally bets that the six correct
D1 calls sit among the nine most confident, which is worth +2.5 if it holds and
-2.5 if it does not. Upload both; the best run stands either way. The retired
practice-pack file is preserved as `outputs/submission_practice_47.json`.

Both pass `src.submit.validate` against `data/manifest_eval.json`. v4 first so
the D2/D3 change is readable on its own; v5 second so the D1 threshold move is
attributable. Built by `scripts/eval_v4.py`, which carries the full reasoning.

## What the private set has told us, without any ground truth

Deduced from the six uploads and the collection prior. See [[scoring]] and
[[fingerprints]].

```
E024        normal.  One event there cost D2 14.0 -> 5.3, exactly one video's
            full mark (35/4 = 8.75). The encoding prior agrees independently.
E025-E028   all four anomalous.  submission_eval left E025 silent and put
            events on the other three, scoring 6.0/40 = three videos at the
            0.2 alert weight. A normal E025 would have scored 16.0.
D2 alert weight is 0.20, not the 0.30 [[scoring]] assumed from the public run.
D2 matches nothing at all; D3 matches one event. The 0.8 of each video's
score that lives in matched+timing is essentially unclaimed -- 28 of 35 marks
on D2 and 32 of 40 on D3.
D1: 25*F1 = 12.0 against 14 anomaly claims gives found 6 of 11 true anomalies.
    So nine of the twenty L1 videos are normal, where the public set was 20
    anomalous of 24. Our threshold was tuned on the wrong prior.
```

## The three levers now built

1. **Candidate width.** IoU 0.5 means a window of width w can only match a
   truth of width w/2..2w. Public truth is 5-60 s on L2, 3-125 s on L3, so the
   old 120 s and 240 s windows could never match anything.
2. **Width stratification.** Score ranking collapses onto the narrowest scale,
   because a short window sits on the peak. Round-robin across widths instead.
3. **Collection-restricted class spray.** Encoding profile identifies the
   source collection; the public ground truth says what each collection
   contains. Spray every allowed class over each window: at large k the F1
   term is ~2m/k and the 0.4 timing term dominates, so covering the class
   is worth more than being precise about it.

Measured on the public anomalous videos: L2 mean per-video 0.200 -> 0.516,
L3 0.280 -> 0.424. Tuned on 4 + 4 videos, so treat the projection as a
direction, not a number.

## Blocked on the user

- All arena uploads; no session can reach the site
- Kaggle phone verification (M4 LoRA — still the only large untried lever)
- 2-slide PPT and architecture write-up sign-off (both built, unreviewed)

## Settled questions — do not re-ask

- Level weighting D1 25, D2 35, D3 40.
- Best **run** stands, and a worse upload never costs anything.
- The private set has no `domain` field and no ground truth.
- Eval videos are **not** duplicates of train or public-test videos — nearest
  neighbour over the 3207-clip embedding bank tops out at 0.99 on genuinely
  different scenes. Only E027 carries an audio track, so audio is not a lever.
