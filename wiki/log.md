# Log

Append-only. Newest at the bottom. Entry format is fixed so the file stays
greppable:

```
## [YYYY-MM-DD HH:MM] <kind> | <title>
```

where `<kind>` is one of `milestone`, `experiment`, `decision`, `finding`,
`blocker`, `session`. Recent activity: `grep "^## \[" wiki/log.md | tail -5`.

---

## [2026-09-05 10:20] session | Read the problem statement PDF

Hackathon is near-real-time video anomaly detection on drone, CCTV and dashcam
footage. Core question: can a small VLM do this reliably in real time. Large
hosted models are development-only and must not be in the runtime path. Build
window 11:00–18:00, demos 18:00–19:00.

Noted a gap: `ground_truth.csv` documents a `level` column as "the task tier
(below)" but the PDF never describes the tiers.

## [2026-09-05 10:30] blocker | Dataset download was 87% incomplete

2.0 GB of 15–17 GB. 384 of 3173 train videos, 31 of 34 test. The six largest
anomaly classes had **zero** videos and `normal` had 10 of 973. Metadata was
complete; only the mp4 payload was missing. Files landed scattered rather than
sequentially, so there was no clean resume point.

## [2026-09-05 10:35] finding | Python 3.14 has no PyTorch wheels

The only Python on the machine was 3.14.7 with nothing but pip installed. torch,
transformers, unsloth and ms-swift all fail on it. Resolved by installing 3.12.

## [2026-09-05 10:45] milestone | Dataset re-downloaded and verified complete

15 GB. 3173/3173 train, 34/34 test. Full integrity audit passed: 0 broken
mappings, 0 orphans, 0 truncated files, 0 bad mp4 headers, 0 missing moov atoms
in a 40-file sample. See [[dataset-audit]].

## [2026-09-05 10:50] finding | Four PRD claims contradicted by the real data

Train has no `level` column; `description_summary` is never blank in train;
every anomaly row already has timestamps so MIL pooling is unnecessary; and
train is entirely short single-event clips while test Levels 2–3 are long and
multi-event. The last one is the structural gap that drives the synthetic
long-sequence design. Also found heavily templated descriptions and at least two
mislabelled `fighting_or_violence` rows.

## [2026-09-05 11:05] milestone | M0 complete — harness, loaders, validator, scorer

Committed `9c919a2`. Loaders verified against the real pack. Scorer runs in
0.36 s.

## [2026-09-05 11:10] finding | Submission is JSON, not CSV

The submission format PDF arrived and invalidated the CSV writer. Also revealed:
`class_name: "normal"` is rejected outright, Level-1 timestamps must be null,
the IoU gate is a hard 0.5, `runtime_metadata` is required per video, there is no
best-of across uploads, and the private set is 28 videos rather than 34.

## [2026-09-05 11:15] milestone | M0b complete — arena JSON format and official scoring

Committed `eb454e2`. `submit.py` and `score.py` both rewritten.
`scripts/test_validation.py` catches all 11 rejection traps with no false
rejects. `scripts/sanity_score.py` confirms the oracle scores exactly 1.000.

## [2026-09-05 11:15] finding | Fragmentation is the dominant Level 2/3 risk

A perfect set of events, correct classes, merely shattered into five slices each
collapses Level 2 from 1.000 to 0.467 and Level 3 from 1.000 to 0.200. Segment
merging matters more than detection accuracy. Thresholds moved accordingly:
`enter` 0.60→0.70, `merge_gap_sec` 2.0→5.0, `min_event_sec` 1.0→2.0.

## [2026-09-05 11:20] decision | Laptop for everything except the LoRA

8 GB VRAM cannot fine-tune a 4B VLM, and the fine-tuning stack is Linux-only.
Everything else runs locally. Only embeddings, SFT frames and the adapter cross
the network — never the 15 GB of video. See [[environment]].

## [2026-09-05 11:25] milestone | M1 — test split encoded

34/34 in 1.3 min at 42x realtime, zero failures. Three fixes were needed:
`AutoImageProcessor` instead of `AutoProcessor` (the latter needs SentencePiece),
pre-resize to 224 during decode (T033 alone held 3.5 GB of raw frames), and
`grab()`/`retrieve()` instead of `read()`. Train split encode started in the
background. Committed `af5af1c`.

## [2026-09-05 11:30] decision | Repo is github.com/SarthakDz/VAD

Remote added, `main` pushed.

## [2026-09-05 11:35] experiment | exp-001 trial head — overall 0.336

18% of the cache, 3 epochs, 0.7 min training. Beats the 0.167 empty baseline.
41.4x realtime. **Every Level 2/3 point comes from `alert`; zero events clear the
IoU 0.5 gate.** Zero false alarms on normal videos. Full config in
[[experiments]].

## [2026-09-05 11:38] milestone | M2 complete — Stage A ships end to end

Committed `082e2d3` and pushed. `head.py`, `dataset_head.py`, `train_head.py`,
`segments.py`, `infer_head.py`. A complete submittable system now exists with no
VLM in it.

## [2026-09-05 11:45] session | Wiki created

Set up `wiki/` plus `CLAUDE.md` as persistent cross-session memory, seeded with
everything established so far.
