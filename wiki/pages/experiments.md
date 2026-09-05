# Experiments

Every run that produces a score goes here with its config, or the number is lost
and gets re-run. Newest last.

All scores are on the **public test set** (34 videos) using `src/score.py` with
the assumed weights documented in [[scoring]]. Reference points: empty
submission 0.167, oracle 1.000.

## Reference points

| strategy | L1 | L2 | L3 | overall |
|---|---:|---:|---:|---:|
| oracle (perfect) | 1.000 | 1.000 | 1.000 | **1.000** |
| empty (all normal) | 0.167 | 0.333 | 0.000 | 0.167 |
| whole-clip accident | 0.479 | 0.133 | 0.200 | 0.271 |
| spam all classes | 0.479 | 0.133 | 0.200 | 0.271 |
| fragmented oracle | 1.000 | 0.467 | 0.200 | 0.556 |

Produced by `scripts/sanity_score.py`. The fragmented-oracle row is the design
constraint for [[architecture]]'s segment logic.

## exp-001 — trial head, partial cache

**2026-09-05 11:35.** Purpose was to verify the pipeline end to end, not to
score well. Run before the train encode finished.

Config:

```
cache          602 of 3207 videos (446 anomaly clips, 176 localised)
train/val      402 / 44 clips
epochs         3
steps/epoch    60
batch size     16
window         256
p_normal       0.5
localised_wt   3.0
lr             1e-3
head           BiGRU hidden 256, 2 layers, dropout 0.2, 2.18M params
segments       enter 0.70, exit 0.45, merge_gap 5.0, min_event 2.0
```

Training: 0.7 min. Measured anomaly frame fraction 0.056 → `pos_weight` 16.9.
Best val `frame_f1` 0.6136, `frame_cls_acc` 0.4439 at epoch 2.

Result:

```
LEVEL 1  0.3750   binary 0.542   class 0.208
LEVEL 2  0.4333   alert 0.750   matched 0.000   timing 0.000
LEVEL 3  0.2000   alert 1.000   matched 0.000   timing 0.000
OVERALL  0.3361

34 videos, 54 events, 82.0 s internal → 41.4x realtime, latency ratio 0.0242
```

**Findings.**

Doubles the empty baseline, and the JSON validates. But **every point at Levels 2
and 3 comes from `alert` — not one predicted event clears the IoU 0.5 gate.** The
head detects *that* something is wrong and not *when*. That is where all Level
2/3 headroom sits.

Both normal Level-2 videos correctly received empty answers, so there are zero
false alarms on normal videos — the conservative thresholds are doing their job.
The Level-2 arithmetic confirms the scorer exactly:
`(2 x 1.0 + 3 x 0.2 + 1 x 0) / 6 = 0.433`.

Level-1 class accuracy of 0.208 is barely above the 1-in-11 floor, which is
expected — the head had seen 6 of 12 classes only sparsely at this cache level.

## Next experiments to run

1. **Full-cache training.** Same config, all 3207 videos, ~15 epochs. Establishes
   the real Stage-A ceiling before any tuning.
2. **Threshold sweep against `matched`, not overall.** Vary `enter`
   (0.5 / 0.6 / 0.7 / 0.8), `merge_gap_sec` (2 / 5 / 10 / 20) and
   `min_event_sec` (1 / 2 / 5). The fragmented-oracle result predicts that
   larger `merge_gap` will help disproportionately. Watch the normal Level-2
   videos: any regression to non-empty answers there costs a full 1.0 each.
3. **`p_normal` sweep.** At 0.5 the synthesised anomaly frame fraction is 0.056,
   which is sparser than the real Level-2 videos (T025 is about 50% anomalous).
   Try 0.3 and 0.2.
4. **Window length.** 256 timesteps is 128 s at 2 fps; the Level-3 videos run to
   629 s. Try 512.
5. **Causal versus bidirectional** — only if a streaming demo is wanted. Expect
   a score drop.
