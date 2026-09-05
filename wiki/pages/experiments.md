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

## exp-002 — full cache, 15 epochs

**2026-09-05 12:00.** Same architecture as exp-001, complete 3207-video cache.

```
clips          2198 anomaly (754 localised) / 946 normal / 25 long normal
train/val      1983 / 215
epochs 15, steps/epoch 400, batch 16, window 256, p_normal 0.5, lr 1e-3
anomaly frame fraction 0.148 -> pos_weight 5.73
```

Training 10.8 min. Best val `frame_f1` **0.9053**, `frame_cls_acc` 0.6894 at
epoch 10; plateaued after. Val is on synthesised sequences, so it is optimistic
relative to the real test set.

```
LEVEL 1  0.7083   binary 0.833   class 0.583
LEVEL 2  0.5136   alert 1.000   matched 0.062   timing 0.130
LEVEL 3  0.2000   alert 1.000   matched 0.000   timing 0.000
OVERALL  0.4740
```

Level 1 nearly doubled over exp-001 (0.375 → 0.708). Level 3 unchanged.

## exp-003 — threshold sweep

210 configs over cached score curves, no GPU. `merge_gap_sec = 20` dominated
every top-15 config; `enter` barely mattered anywhere from 0.4 to 0.8, which is
the signature of a saturated score curve rather than a well-calibrated one.

Best: `--enter 0.6 --exit 0.3 --merge-gap 20 --min-event 1` → overall 0.497.

## exp-004 — timestamp drift fix ← current best

**A correctness bug, not a tuning change.** Per-video inspection showed T028
predicting `32-37, 94-99, 157-162, 219-224` against ground truth
`30-35, 90-95, 150-155, 210-215` — offsets of +2, +4, +7, +9, growing linearly.

`frames.sample()` keeps every `step = n_frames // want` frames, an integer
floor, so the kept frames span slightly less than the whole video. But
`timestamps()` spread them with `linspace` across the full `duration_sec`. For
T028 the real spacing is 12/25 = 0.480 s and the assumed spacing was 0.501 s —
a 4.4% drift that silently pushed every boundary late and failed the IoU gate.

Fixed by storing `frame_step` in `VideoMeta` and deriving
`timestamps = arange(n) * frame_step / native_fps`. The 3207 cached meta files
were backfilled without re-encoding, since every input was already stored.

T028 afterwards: `30.2-35.0, 90.2-95.0, 150.2-155.0, 210.2-215.0`.

```
LEVEL 1  0.7083   binary 0.875   class 0.542
LEVEL 2  0.5972   alert 1.000   matched 0.250   timing 0.236
LEVEL 3  0.2704   alert 1.000   matched 0.029   timing 0.147
OVERALL  0.5253

41.7x realtime, latency ratio 0.0240
```

Level-2 `matched` quadrupled, 0.062 → 0.250.

## exp-005 — adaptive coverage cap (negative result)

Four Level-2/3 videos (T027, T031, T032, T034) saturate: the head scores
essentially every frame above any fixed threshold, producing one segment
spanning the whole video, which fails the IoU gate outright — T034 claims 377 s
for a 10 s event, IoU 0.027.

Added `max_coverage` to `segments.extract`: when coverage exceeds the cap, fall
back to a per-video quantile. Tested 0.7 / 0.5 / 0.35 / 0.25.

**No improvement at any value, and 0.25 made Level 3 worse** (0.270 → 0.200).
The curve is saturated *near 1.0*, so the quantile cut lands arbitrarily rather
than on the true event. This is not a thresholding problem — the head has no
temporal signal on these videos at all.

The code is kept, documented, and defaults to disabled. Note that saturation on
the short Level-1 clips is correct behaviour, so the cap is never applied there.

## Where the remaining Stage-A error actually is

Per-video inspection of all ten Level-2/3 videos found three separate failures,
worth keeping distinct:

**1. Confidently wrong class, perfect timing.** T025 has six accidents; the head
finds all six intervals almost exactly and labels every one
`wrong_way_driving` — with 98.2% of the class mass, and `traffic_accident` not
even in its top four. Six of the sixteen Level-2 ground-truth events are lost to
this single error. No threshold or vote change can fix a confidently wrong
classifier. **This is exactly the case Stage B is for: arbitrate the class on a
segment whose boundaries are already right.**

**2. No temporal signal on saturated videos.** T027, T031, T032, T034 as above.
T032 and T034 are the 1.88 fps 896x448 loitering source — the class the audit
flagged as a probable single-benchmark rip with 300 near-identical clips, so the
head plausibly learned "this source is anomalous" rather than what loitering
looks like. See [[dataset-audit]].

**3. Over-fragmentation on genuinely multi-event video.** T033 produces 15
segments for 2 ground-truth events, with the class flapping between
`traffic_accident`, `wrong_way_driving`, `fire` and `fighting_or_violence`.
T026 produces 12 for 4. Raising `merge_gap` to 20 helped but did not solve it;
the `_split_by_class` blip guard is not aggressive enough on long videos.

Failure 1 is the largest and is Stage B's job. Failure 3 is worth another pass
at `min_run`. Failure 2 needs different training data and probably cannot be
fixed today.

## exp-006 — Stage B with Qwen3-VL-2B (negative result)

**2026-09-05 12:45.** Zero-shot re-labelling of Stage-A segments. Model loads in
15.5 s and occupies **4.26 GB** VRAM, 2.3-3.9 s per call at 6 frames.

Targeted test on segments whose ground truth is known:

```
T025 20-40s   head wrong_way_driving   gt traffic_accident   vlm traffic_congestion      MISS
T025 60-80s   head wrong_way_driving   gt traffic_accident   vlm traffic_congestion      MISS
T028 30-35s   head traffic_accident    gt traffic_accident   vlm vehicle_blocking_traffic MISS
T027 65-125s  head traffic_congestion  gt traffic_congestion vlm traffic_congestion      OK
```

**It is worse than the head, and it actively broke T028**, which Stage A had
right. The explanations are fluent and confidently wrong — *"A blue car is seen
blocking the road in the center of the lane"* for a genuine collision.

Ablations on four accident segments across T025 and T028:

```
6 frames,  shortlist of 6   0/4
12 frames, shortlist of 6   0/4
20 frames, shortlist of 6   0/4
12 frames, all 11 classes   0/4
```

Frame count does not help, so this is not the sampling hypothesis (an accident
lasts about a second, and 6 frames over 20 s samples every 3.3 s — plausible,
but ruled out). Shortlist size does not help either. **Qwen3-VL-2B simply cannot
identify traffic accidents in this footage.** It reads every accident as
congestion or a blocking vehicle.

Consistent with [[prior-art]]: the AI City Track 3 leaderboard is dominated by
Qwen3-VL-**8B**, and the top score there was 0.6788 on a task that did not even
score temporal localisation. 2B is well below the size that worked.

Escalating to Qwen3-VL-4B in 4-bit (8.89 GB will not fit 8 GB VRAM in bf16).

**Fusion policy note.** `fuse.py` never lets a failed or out-of-shortlist VLM
answer through — the head's class stands. But this experiment shows that is not
enough protection: the VLM returned a *valid, in-shortlist, wrong* answer and
overwrote a correct one. If 4B is also mixed, Stage B must be gated rather than
applied blindly — for example, only overriding on segments where the head's
class mass is below a threshold, accepting that this cannot fix T025, where the
head is 98.2% confident and wrong.

## exp-007 — Stage B with Qwen3-VL-4B in 4-bit (negative, decisive)

**2026-09-05 13:10.** 4-bit NF4, loads in 23.9 s, **2.91 GB** weights, 6.89 GB
peak VRAM, 3.8-9.5 s per call at 8 frames.

Probe segments: **VLM 1/6 correct, head 3/6.** It broke both T028 segments the
head had right, and missed all three T025 accidents exactly as 2B did.

Full run, 59 calls over 472 frames, 39 of 57 segments relabelled, 0 parse
failures:

```
              head only        head + VLM
LEVEL 1         0.7083           0.6875
LEVEL 2         0.5972           0.5347
LEVEL 3         0.2704           0.2704
OVERALL         0.5253           0.4976
realtime          41.7x             5.6x
```

In leaderboard units the D2 collapse is worse than the average suggests: found
drops 4/18 to 1/18 and false alarms rise 16 to 19.

**Stage B is dropped.** Two model sizes, consistent direction, and a 7.4x
latency cost for a lower score. `vlm.py` and `fuse.py` stay in the tree behind
`--vlm`, off by default, because the negative result is worth showing and the
motion-crop and shortlist code is reusable if a fine-tuned model appears.

Note the runner-up on the live leaderboard runs `qwen3vl4b-lora-finetuned` and
scores 51.1 — so a 4B Qwen *can* work here, but evidently only fine-tuned, not
zero-shot. That is M4, and it needs Kaggle.

## exp-008 — retune for the real objective ← current submission

Once the real weighting was known (25/35/40) and the leaderboard showed
precision dominating, the sweep was rerun on weighted marks with false alarms as
tiebreak. 1188 configs.

Marks plateau near 49 across a wide range, so the choice became false-alarm risk
rather than modelled score. Chosen config trades 0.8 modelled marks to cut false
alarms by a quarter:

```
--enter 0.92 --exit 0.30 --merge-gap 20 --min-event 3
```

```
                     P      R    found   FA        (previous config)
D1 Clear event      64%    45%    9/20    5              53% 10/20  9
D2 When it happens  25%    22%    4/18   12              20%  4/18 16
D3 Long context      9%    12%     1/8   10               6%  1/8  17
                            41 events, FA 27            57 events, FA 42

LEVEL 1 0.6458   LEVEL 2 0.5972   LEVEL 3 0.2787
marks ~49/100    41.7x realtime, latency ratio 0.0240
```

Precision improves on all three difficulties and total false alarms fall 42 to
27. Modelled marks put us near the current second place (47.6) and far below the
leader (92.1).

**The honest gap:** we match the leader's Difficulty-2 recall exactly — 4 of 18
— and lose entirely on precision, 12 false alarms against his 0. Recall is not
the problem and never was.
