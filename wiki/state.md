# State — where the work stands

**Last updated:** 2026-09-05 14:35

## Position

Stage A is finished, tuned and validated. Stage B was built, measured, and
**dropped as a negative result**. The submission has been ready for some time.

**The only thing blocking us is the arena upload endpoint, which rejects every
file we send — including its own unmodified starter template.** See the
blocker section below; it is not a problem with our output.

```
outputs/submission_from_template.json   <- upload this
34/34 videos, 41 events, 48.2 modelled marks /100
41.7x realtime, latency ratio 0.0240
passes all 11 validation traps and every rule on the arena FIELD RULES page
```

Modelled marks put us around the current second place (47.6) and far below the
leader (92.1). See [[scoring]] for the real weighting and [[experiments]] for
every run.

## THE BLOCKER — arena upload fails

Every upload returns:

> Could not save that submission: **Too many parameter values were provided**

It is **not** a size or shape problem on our side. The bisection, in order:

| what was sent | leaves | result |
|---|---:|---|
| full submission, 68 `model_runtimes` rows | 537 | parameter error |
| merged to 34 rows | 401 | parameter error |
| `model_runtimes: []` | 265 | parameter error |
| all 34 videos, **zero events** | 141 | parameter error |
| 3 videos, template shape | 23 | parameter error |
| **1 video, empty events, 4 leaf values** | **4** | **parameter error** |
| `model_runtimes` key omitted entirely | 259 | reached the validator ✓ |
| 34 rows x 7 fields, matching the doc example | — | parameter error |

A 4-leaf JSON document cannot exceed any bind-parameter limit, so this is
server-side.

The two validators also contradict each other: `model_runtimes: []` triggers the
parameter error, while omitting the key returns *"model_runtimes must be an
array (use [] if not applicable)"* — advice that leads straight back into the
first failure.

**Decisive fact:** the official starter template, downloaded from the Benchmark
tab on 2026-09-05 at 13:15 and saved to `data/submission_template_official.json`,
itself ships `"model_runtimes": []` on all 34 videos. So empty is the intended
value and the error message is misleading.

### Next actions on the blocker

1. Upload `data/submission_template_official.json` **unmodified**. If their own
   file fails, the diagnosis is finished — report it and stop probing.
2. Upload `outputs/submission_from_template.json`. This is their downloaded file
   with only two changes: the `events` arrays, and the three runtime scalars as
   integers. It is the first file we have sent that is their artifact rather than
   one written from scratch, so it bypasses any type, key-order or
   field-emission difference.
3. Fallback `outputs/submission_from_template_int.json` — as above with integer
   event timestamps too. Scores 48.3, marginally better.

Rejections **do not consume an attempt**, so probing is free.

### Message to send if step 1 fails

> Your own starter template, downloaded from the Benchmark tab and uploaded back
> unmodified, fails with "Could not save that submission: Too many parameter
> values were provided". Account: sarthakdhaigude5337@gmail.com

Also worth asking what shape the leader's submissions are — 27 successful runs
means someone found a form that works, and that is the fastest route to
unblocking.

## Next action once unblocked

Upload immediately, then iterate. **The submission page states "your best run
stands, so a worse attempt never costs you"**, which reverses the format PDF's
"no best-of" — uploading is now zero-risk and a live score is the only way to
calibrate the assumed weights in [[scoring]].

After that, in value order:

1. **M4 — LoRA fine-tune on Kaggle.** The current second place runs
   `qwen3vl4b-lora-finetuned`. Our zero-shot VLM failed at both 2B and 4B, so
   fine-tuning is the demonstrated path, and it is the only untried lever with
   real headroom. Needs Kaggle phone verification.
2. **Cut false alarms further.** We match the leader's Difficulty-2 recall
   exactly (4/18) and lose entirely on precision, 12 false alarms against his 0.
3. **Deliverables** — 2-slide PPT (stated high weightage), architecture
   write-up, both still unstarted.

## Blocked on the user

- The uploads above; no session can reach the arena
- **Kaggle phone verification** — GPU stays locked, needed for M4
- 2-slide PPT and architecture write-up at the end

Full list and paste-ready questions in [[open-questions]], though the leaderboard
has since answered the level-weighting question outright.

## Recent decisions

- **Stage B dropped.** Qwen3-VL-2B scored 0/4 on probe segments and 4B in 4-bit
  scored 1/6 against the head's 3/6, lowering the full run to 0.4976 from 0.5253
  at 7.4x the latency. Kept behind `--vlm`, off by default.
- **Retuned for precision, not recall**, once the leaderboard showed false
  alarms dominate. False alarms 42 -> 27.
- Bidirectional GRU, synthetic long training sequences, no MIL — unchanged, see
  [[architecture]].
