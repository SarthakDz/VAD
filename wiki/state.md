# State — where the work stands

**Last updated:** 2026-09-05 14:40

## Position

**Submitted and scored: 47.0 / 100** (D1 12.9/25, D2 22.6/35, D3 11.5/40).
The upload blocker is resolved — `outputs/submission.json` went through.

Stage A is complete and **at its ceiling**. Stage B was built, measured and
dropped. Three separate attempts to improve on 47.0 have all failed.

```
outputs/submission.json     <- the standing score, do not overwrite casually
34/34 videos, 41 events, 41.7x realtime, latency ratio 0.0240
outputs/head.pt             <- the head that produced it
```

Uploading is **zero-risk**: the submission page says *"your best run stands, so
a worse attempt never costs you"*, which reverses the format PDF. Experimental
uploads cost nothing.

## Use `calibrated.py`, not `score.py`

The real arena result let us reverse-engineer the marks formula.
`src/calibrated.py` reproduces it **exactly — 47.0 predicted, 47.0 actual**,
each difficulty within 0.4 marks. It is the only scorer to tune against.

Key corrections it encodes (see [[scoring]]):
- **D1 is F1-based**, not the PDF's `0.5*binary + 0.5*class`, so D1 false alarms
  cost marks directly
- **D2 weights are ~(0.3, 0.4, 0.3)**, not the assumed (0.2, 0.5, 0.3)

## Everything tried since 47.0 has failed

```
threshold sweep, 1800 configs        +0.1
window 512                           -3.5
organiser label corrections          -5.2
Stage B VLM, Qwen3-VL 2B and 4B      worse at both sizes, 7.4x slower
```

Full detail and caveats in [[experiments]] exp-010 through exp-012. **Do not
re-run these.** Stage A needs a different model, not more tuning.

## Where the remaining marks are

```
D1  12.9 / 25    found 9/20, 5 false — all pair confusions
D2  22.6 / 35    the healthiest difficulty
D3  11.5 / 40    28% — worst, and the largest pool
```

The arena's own guidance: *"You are flagging more events than are there.
Cutting false alarms will raise your marks more than finding extra events
will."* Precision 34%, recall 30%, **27 false alarms**.

18 of the 22 temporal false alarms come from three videos — T026 (7, matched
0/4), T033 (7, matched 1/2), T025 (4, matched 0/6). On T026 we matched nothing,
so predicting less there costs no recall at all.

D1's 5 false alarms are all pair confusions: `fire`↔`smoke` twice (a pure swap),
`fighting_or_violence`→`loitering` twice, `road_spill`→`traffic_accident`.

## Next action — decision pending with the user

Asked at 14:35, not yet answered. Two options:

1. **M4 — LoRA fine-tune on Kaggle.** The only lever with real headroom. Second
   place on the live leaderboard runs `qwen3vl4b-lora-finetuned` at 51.1, and our
   zero-shot VLM failed at both 2B and 4B, so fine-tuning is the demonstrated
   path. **Blocked on Kaggle phone verification.** Plan in [[milestones]].
2. **Deliverables** — the 2-slide PPT (explicitly stated high weightage) and the
   architecture write-up. Both unstarted, neither depends on the score.

Recommendation given: if Kaggle is not verified, do the deliverables. A 47.0
with a sharp write-up beats a 47.5 with no slides. Material is strong — a 42x
realtime cascade, a real timestamp bug found by inspection, two documented
negative results, and a scorer reverse-engineered from a single submission.

## Blocked on the user

- **Kaggle phone verification** — GPU stays locked, needed for M4
- All arena uploads; no session can reach the site
- 2-slide PPT and architecture write-up sign-off

## Settled questions — do not re-ask

- **Level weighting** is D1 25, D2 35, D3 40. Read off the leaderboard.
- **Best run stands.** The PDF's "no best-of" is wrong.
- **`manifest.json`** is `{videos:[{video_id, level, domain, duration_sec}]}` and
  parses unmodified; 34 videos, levels 24/6/4.
- **The arena scores the same public 34 videos we score locally.**

## Recent decisions

- **Keep the original labels.** The organisers' `wrong_way_driving` correction
  costs 5.2 marks despite being principled; `apply_corrections=False` is the
  default path back. See [[experiments]] exp-012 for the caveats.
- **Stage B off by default**, kept behind `--vlm` because the negative result is
  worth showing on the slides.
- Bidirectional GRU, synthetic long training sequences, no MIL — see
  [[architecture]].
