# State — where the work stands

**Last updated:** 2026-09-05 11:40

## Position

Milestone **M2 complete and committed**. A full Stage-A pipeline runs end to end
and produces a validated arena submission with no VLM in it. See
[[milestones]] for the plan and [[experiments]] for scores.

Best score so far, on the public test set, from a deliberately undertrained
trial head (18% of the cache, 3 epochs):

```
LEVEL 1  0.375    LEVEL 2  0.433    LEVEL 3  0.200    OVERALL  0.336
empty baseline 0.167   oracle 1.000
41.4x realtime, latency ratio 0.0242
```

## Running right now

`src.encode --split train` in the background, writing to `cache/emb/`.
At 11:38 it stood at **1393 of 3207**, moving ~111/min.

**Verify before trusting this line** — count `cache/emb/*.npy` and compare to
3207 (3173 train + 34 test). If it is short and no job is running, restart:

```bash
./.venv/Scripts/python.exe -m src.encode --split train
```

It resumes automatically; anything already cached is skipped.

## Next action

Once the cache is complete, in this order:

1. **Train the head properly** — full cache, ~15 epochs.
   `./.venv/Scripts/python.exe -m src.train_head --epochs 15 --out outputs/head.pt`
2. **Sweep segment thresholds against `matched` specifically**, not overall
   score. This is the highest-value remaining work — see the diagnosis below.
3. Then M3 (VLM zero-shot) if time allows.

## The diagnosis that should drive the next hour

At Levels 2 and 3 the score decomposes as alert / matched / timing. Currently:

```
LEVEL 2  alert 0.750  matched 0.000  timing 0.000
LEVEL 3  alert 1.000  matched 0.000  timing 0.000
```

**Every point is coming from `alert`. Not one predicted event clears the IoU 0.5
gate.** The head knows *that* something is wrong and not *when*. All Level 2/3
headroom is in segment boundary placement, which means threshold sweeping and
[[architecture]]'s merge logic — not more training epochs.

One genuinely good sign: both normal Level-2 videos received empty answers, so
there are zero false alarms on normal videos, the most expensive error class.

## Blocked on the user

These cannot be done from inside the session. Full list and framing in
[[open-questions]].

- **Arena site URL and login** — not in any provided PDF. Without it nothing can
  be submitted at all.
- **`manifest.json`** from the arena Benchmark tab, saved to `data/manifest.json`.
  The parser in `src/submit.py` accepts four plausible shapes but has never seen
  the real file. Needs a two-minute check once it exists.
- **Level 2/3 component weights** — currently guessed. See [[scoring]].
- **Kaggle phone verification** — GPU stays locked until done, needed for M4.
- 2-slide PPT and architecture write-up at the end. Stated high weightage.

## Recent decisions

- Bidirectional GRU, not causal. The arena scores wall-clock time, not
  causality, and every input is a finished file.
- Synthesise long multi-event training sequences by concatenating clip
  embeddings. The training set has no long-form anomaly footage at all; Levels
  2 and 3 are entirely long-form. See [[dataset-audit]].
- Skip the MIL top-k pooling the original PRD planned. Every anomaly row already
  carries timestamps, so supervision is dense everywhere.
