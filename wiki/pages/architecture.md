# Architecture

Three stages. Stage A is the workhorse and is complete; Stage B is a strict
upgrade; Stage C is glue. The design goal is that a valid submission exists
early and every later step can only improve it.

```
video.mp4
  |
  A1  decode + uniform sample @ 2 fps        src/frames.py
  A2  frozen SigLIP encoder                  src/encode.py    -> cache/emb/{id}.npy
  A3  temporal head (BiGRU)                  src/head.py      -> per-second scores
  |
  C1  hysteresis + merge + class split       src/segments.py  -> candidate segments
  |
  B   VLM on candidate segments only         src/vlm.py       (NOT BUILT YET)
  |
  C2  fuse and emit                          src/infer_head.py -> submission JSON
```

## Why this order

Stage A alone produces a valid Level 1 *and* Level 2/3 submission with no VLM
anywhere. If the LoRA never converges, the system still ships. The cost story is
also legible: the frozen encoder runs at hundreds of fps, and the VLM would touch
only candidate segments — target under 5% of frames.

## Stage A — decode and encode

**`src/frames.py`.** Decode dominates M1 wall-clock, so the read loop matters.
OpenCV's `read()` fully decodes every frame; at 2 fps on 30 fps footage we keep 1
in 15, so 14 of 15 decodes would be wasted. The code uses `grab()` to advance the
demuxer without converting to a numpy array and only calls `retrieve()` on kept
frames. Measured result: **42x realtime** on the public test set.

Seeking via `CAP_PROP_POS_FRAMES` is deliberately not used — unreliable across
the codec mix (CCTV, dashcam, drone all differ) and can silently land on the
wrong frame.

Frames are pre-resized to 224 during decode. This is not just speed: T033 alone
held 3.5 GB of raw 720p frames, and T029 at 1080p would have exhausted RAM.
SigLIP squashes to 224x224 anyway, so nothing is lost.

**`src/encode.py`.** `google/siglip-base-patch16-224`, frozen, never fine-tuned.
Writes `cache/emb/{video_id}.npy` as fp16 `(T, 768)` plus
`cache/meta/{video_id}.json` carrying `duration_sec`, `decode_sec` and
`encode_sec`. Reruns skip anything already cached.

Use `AutoImageProcessor`, **not** `AutoProcessor` — the latter also builds the
SigLIP text tokenizer and hard-fails on missing SentencePiece.

Videos are processed longest-first so the 800 MB files surface OOM or codec
problems in the first minute rather than the last.

## Stage A — temporal head

**`src/head.py`.** LayerNorm → Linear → GELU projection, then a 2-layer
bidirectional GRU (hidden 256), then two output heads: a per-timestep anomaly
logit and per-timestep 12-class logits. **2.18M parameters.**

Bidirectional by default. The arena scores wall-clock processing time, not
causality, and every input is a finished file, so looking ahead is allowed and
materially improves boundary placement — which is what the IoU 0.5 gate actually
rewards. `--causal` switches to unidirectional for a streaming demo.

Inference is chunked at 4096 timesteps with 128 timesteps of discarded overlap on
each side, so a 20-minute video cannot exhaust VRAM and chunk boundaries do not
develop seams.

## Stage A — training data synthesis

**`src/dataset_head.py`.** This is the piece that addresses the train/test
structural gap documented in [[dataset-audit]]: training clips are short and
single-event, but Levels 2 and 3 are long, multi-event and multi-class.

`SyntheticSequences` assembles random windows (default 256 timesteps ≈ 128 s) by
concatenating clip embeddings — anomaly clips with their intervals mapped into
window coordinates, interleaved with normal filler drawn preferentially from the
8 real long normal videos. Concatenation happens in **embedding space**, so it
costs microseconds and no video is ever re-decoded.

Anomaly clips are sampled inversely to class frequency so `traffic_accident`
(565 clips) does not drown out `fire` (77).

`localised_weight` (default 3.0) upweights timesteps from the ~649 clips whose
event does not span the whole clip — the only real boundary supervision that
exists.

**`src/train_head.py`.** Dense supervision: BCE on the anomaly logit plus
cross-entropy on class logits, both weighted per timestep. `pos_weight` is
**measured** from the actual anomaly frame fraction rather than guessed (came out
at 16.9 for a fraction of 0.056). AdamW with OneCycleLR. Validation clips are
held out whole and stratified by class.

## Stage C — segments

**`src/segments.py`.** This module decides the Level 2/3 score more than
detection accuracy does — see the fragmented-oracle result in [[scoring]].

The bias is deliberately toward few, long, merged segments:

- **Hysteresis** — a high `enter` threshold to start a segment, a lower `exit` to
  continue it, so one event does not flicker into forty
- **Merge** anything separated by less than `merge_gap_sec`
- **Drop** anything shorter than `min_event_sec`
- **Split on sustained class change** with a `min_run` guard, so T026-style
  multi-class videos are handled without per-frame jitter fragmenting a span

Class within a segment is a score-weighted vote that ignores the `normal` column —
inside a span the question is which anomaly it is, not whether it is one.

Current defaults, chosen because false alarms are more expensive than misses:

```yaml
enter_threshold: 0.70
exit_threshold:  0.45
merge_gap_sec:   5.0
min_event_sec:   2.0
```

## Stage C — inference and submission

**`src/infer_head.py`.** Loads cached embeddings, scores, extracts segments,
converts to arena events (Level 1 collapses to a single best event with null
timestamps), and writes a validated submission.

Runtime accounting is **real, not invented**. Decode and encode times are
recorded per video at encode time and stored in `cache/meta`; head inference is
timed at inference. `end_to_end_internal_time_ms` is their honest sum, which is
what the arena asks for — decoding, preprocessing, inference and postprocessing,
excluding model load.

A video with no cached embedding still gets an entry with an empty answer. Never
drop a video.

## Stage B — not built yet

Planned: `Qwen/Qwen3-VL-4B-Instruct`, LoRA fine-tuned, invoked only on Stage A's
candidate segments with a constrained JSON output and a **class shortlist** from
the confusable groups in `src/labels.py` rather than 12 flat options.

Two notes carried forward for whoever builds it. The VLM must read **original
frames cropped to the motion region**, not the 224px embeddings — small distant
objects in 1080p drone footage are already gone by encode time. And the motion
mask idea from the Cerberus paper (frame differencing, dilate, crop or dim
outside) is roughly 20 lines with no released code, so implement it behind a
config flag and A/B it.
