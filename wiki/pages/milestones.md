# Milestones

Order is deliberate: a valid submission must exist early, and every later step is
a strict upgrade that cannot take the system down. Current position always lives
in `wiki/state.md`, not here.

## M0 — Harness ✅ complete

`labels.py`, `io_dataset.py`, `submit.py`, `score.py`, `configs/default.yaml`,
`scripts/00_smoke.sh`.

Accepted when the loaders parsed the real pack (3173 train / 34 test rows, 0
missing files, 0 invalid class strings), baselines validated and scored, and the
scorer ran in well under the 10 s target. It runs in **0.36 s**.

Originally written for a CSV submission. Rewritten as **M0b** when the submission
format PDF arrived — see [[scoring]]. The rewrite added `scripts/test_validation.py`
(11 rejection traps, all caught, no false rejects) and `scripts/sanity_score.py`
(oracle scores exactly 1.000).

## M1 — Embeddings ✅ complete for test, running for train

`frames.py`, `encode.py`.

Test split: 34/34 encoded in 1.3 min at **42x realtime**, zero failures. Train
split runs in the background at ~111 videos/min.

Three fixes were needed and are worth remembering: `AutoImageProcessor` instead of
`AutoProcessor`, pre-resize to 224 during decode, and `grab()`/`retrieve()`
instead of `read()`. All three are explained in [[architecture]].

## M2 — Temporal head → first real submission ✅ complete

`head.py`, `dataset_head.py`, `train_head.py`, `segments.py`, `infer_head.py`.

Accepted when the pipeline produced a validated submission beating the empty
baseline with non-empty timestamps at Levels 2 and 3. A trial head on 18% of the
cache scored **0.336 overall against a 0.167 baseline**.

**At this point a complete, submittable system exists with no VLM. It is
committed and pushed. Everything after this is upgrade.**

Remaining work inside M2: train on the full cache, then sweep segment thresholds
against the `matched` component. See [[experiments]].

## M3 — VLM zero-shot ❌ built, measured, dropped

`vlm.py`, `fuse.py`. Stock `Qwen3-VL-4B-Instruct`, constrained JSON output,
shortlist prompting from `labels.CONFUSABLE_GROUPS`, running only on Stage A's
candidate segments.

**Outcome: head-only wins, so Stage B is off.** Qwen3-VL-2B scored 0/4 on probe
segments at every frame count and shortlist size; Qwen3-VL-4B in 4-bit scored
1/6 against the head's 3/6 and lowered the full run to 0.4976 from 0.5253 at
7.4x the latency. Both relabelled correct predictions to wrong ones. Kept behind
`--vlm`, off by default — the negative result is worth showing on the slides,
and the motion-crop and shortlist code is reusable if a fine-tuned model appears.
See [[experiments]] exp-006 and exp-007.

Optional 45-minute timebox in parallel: TAU-R1's Qwen3-VL-2B classifier from
HuggingFace (MIT licensed) as a second zero-shot baseline. It was trained on
roadside CCTV in one US town with a different label set, so treat it as a free
data point, not a foundation. Drop it if it is not producing output in 45 minutes.

## M4 — LoRA fine-tune ⬜ not started, needs Kaggle — now the highest-value work

The current second place on the live leaderboard runs `qwen3vl4b-lora-finetuned`
and scores 51.1. Zero-shot failed at both 2B and 4B, so fine-tuning is the
demonstrated path and the only untried lever with real headroom.

`build_vlm_sft.py` → `train.jsonl` / `val.jsonl` → ms-swift **or** Unsloth →
merged adapter → swap into `vlm.py`.

**Pick one framework and do not switch mid-day.**

Key input from [[dataset-audit]]: dedupe the SFT set to the ~526 distinct
(class, description) pairs and upweight the five genuinely descriptive classes.
The raw 3173 rows contain 300 byte-identical loitering targets.

Reference command from the organisers' primer:

```bash
swift sft --model Qwen/Qwen3-VL-4B-Instruct \
  --dataset train.jsonl --val_dataset val.jsonl \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 --learning_rate 1e-4 \
  --num_train_epochs 1 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 --gradient_checkpointing true \
  --max_length 4096 --output_dir output
```

`IMAGE_MAX_TOKEN_NUM` and `FPS_MAX_FRAMES` are the memory and latency dials. For
Unsloth: `finetune_vision_layers=False`, `finetune_language_layers=True`, `r=16`,
`lora_alpha=16`, `target_modules="all-linear"`, `UnslothVisionDataCollator` with
`train_on_responses_only=True`, and **build the dataset with a list
comprehension, not `dataset.map()`** — mapping breaks on multi-image samples.
For TRL: `max_length=None` in `SFTConfig` or truncation silently cuts image
tokens.

Accept when the fine-tuned model beats zero-shot on class accuracy and the
adapter loads locally. If it does not beat zero-shot, keep zero-shot and move on.

## M5 — Tune, benchmark, freeze ⬜ not started

Sweep thresholds, run the benchmark table, write results. **Freeze at 17:30.**

## Final deliverables — separate from the score

- Validated submission covering every video
- Code repository URL — `https://github.com/SarthakDz/VAD` ✅
- Architecture write-up, link or PDF/HTML up to 25 MB
- **2-slide PPT, stated to carry high weightage.** Prefer visuals: model choices,
  experiments tried, sampling strategy, temporal logic, what failed, what worked.
- Benchmark table: fps, VLM calls per video-minute, % of frames the VLM touched,
  peak VRAM, parameter count

## Things explicitly not to do

- Do not clone a research repo as a base. VadCLIP is hardwired to
  UCF-Crime/XD-Violence with no custom-data path; VAU-R1's contribution is GRPO
  RL training that will not converge on a T4 in an afternoon. The one exception
  is copying VAU-R1's temporal-grounding prompt format and IoU reward function as
  read-only reference for Stage B prompt design.
- Do not put a hosted model in the runtime path. Gemini Flash and NVIDIA NIM are
  for generating SFT data and comparison baselines only.
- Do not feed full frames to the VLM. Crop to the motion region.
- Do not build a live-streaming system, NVR or event-dedup UI. The submission is
  a file.
- Do not skip temporal localisation. It is the differentiator.
