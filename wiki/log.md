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

## [2026-09-05 11:50] milestone | Train split encoded — cache complete

3207/3207 embeddings cached (3173 train + 34 test). Full head training started.

## [2026-09-05 11:52] finding | Prior art from the organisers' deck and the AI City paper

`research paper.pdf` is The 10th AI City Challenge, arXiv:2608.17044. Its Track 3
(TAR) is the closest published benchmark to this task. Top score 0.6788, Qwen3-VL
family dominant, a GPT-based entry placed 16th. The paper concludes there is
"a shift from simple VLM prompting toward agentic pipelines that first extract
visual evidence, then match it to a task-specific answer format" — which is the
Stage A / Stage B cascade already built.

**Track 3's official mean excludes temporal localization; ours does not.** So
their numbers are not comparable and the published methods were not optimised for
the dimension we are scored hardest on. Independently confirms the
fragmented-oracle conclusion in [[scoring]].

Also surfaced `nvidia/Cosmos-Embed1-448p-anomaly-detection`, a purpose-built VAD
embedding model with the same 768 dims as our SigLIP encoder — attractive but
clip-level not frame-level, and licensed "other". Full notes in [[prior-art]].

## [2026-09-05 12:15] finding | Real manifest.json received and validated

Shape is `{schema_version, videos: [{video_id, level, domain, duration_sec}]}`.
`load_manifest` parsed it unmodified. **34 videos, levels 24/6/4** — so the arena
runs the public test set, not a separate 28-video set as the format PDF's example
implied. Zero level mismatches and zero duration deltas over 1 s against our own
`test/ground_truth.csv` and decoded metadata, which independently confirms the
dataset audit and the timestamp fix.

Because the arena scores the same 34 videos we score locally, **our local scorer
should track the live one closely** — the first real run can be used to calibrate
the assumed Level 2/3 weights in [[scoring]].

`domain` is empty on all 34; likely populated on the private set.

Also processed the arena **starter template**, which is a different file: entries
are `{video_id, events, runtime_metadata}` with **no `level`**, so it cannot drive
output. Added `load_template()` for the ID list and made `load_manifest` detect a
template and say so. The template also revealed an undocumented field,
`run_metadata.max_parallel_videos`, now emitted.

## [2026-09-05 12:16] milestone | First manifest-driven submission ready

`outputs/submission.json` — 34/34 videos, 57 events, 17.2 KB of the 5 MB limit,
0 events past their manifest duration, all 11 validation traps pass.
Overall 0.5253 locally.

## [2026-09-05 12:45] experiment | Qwen3-VL-2B is not good enough for Stage B

0/4 on known accident segments at 6, 12 and 20 frames, and with both a
6-class shortlist and all 11 classes. It relabelled a correct T028 prediction to
`vehicle_blocking_traffic`. Stage B built and wired (`vlm.py`, `fuse.py`,
`--vlm` flag on `infer_head`), pipeline verified to be unchanged with it off.
Escalating to Qwen3-VL-4B in 4-bit. Full detail in [[experiments]].

## [2026-09-05 13:30] finding | Live leaderboard reveals the real scoring weights

D1 out of 25, D2 out of 35, **D3 out of 40** — levels are not equally weighted,
and Level 3 is both the most valuable and our weakest. Earlier tuning optimised
an equal-weighted objective and was aiming at the wrong target.

More important: the leaderboard shows **precision and false alarms dominate**.
The leader scored 85% of D2 having found only 4 of 18 events, on zero false
alarms; the runner-up found 5 and scored less with 8 false alarms. We had 42
false alarms. Full analysis in [[scoring]], new module `src/leaderboard.py`
reports P/R/found/FA in the arena's own units.

## [2026-09-05 13:35] experiment | Stage B dropped after 4B also failed

Qwen3-VL-4B in 4-bit scored 1/6 on probe segments against the head's 3/6 and
lowered the full run to 0.4976 from 0.5253 at 7.4x the latency. Two model sizes,
same direction. Kept behind `--vlm`, off by default. See [[experiments]].

## [2026-09-05 13:40] milestone | Final Stage-A submission retuned for precision

`--enter 0.92 --exit 0.30 --merge-gap 20 --min-event 3`. False alarms 42 -> 27,
precision up on all three difficulties, 41 events, ~49 modelled marks.
`outputs/submission.json` validated and ready to upload.
