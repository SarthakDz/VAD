# AHC Visual Intelligence Hackathon — near-real-time video anomaly detection

A three-stage cascade that watches a video and answers three questions: *is
something wrong*, *what is it*, and *when did it happen*. It runs **41.7× faster
than realtime on a laptop RTX 4060** and puts **no hosted model in the runtime
path**, which the organisers forbid.

```
decode @ 2 fps  ->  frozen SigLIP  ->  2.18M-param BiGRU  ->  hysteresis + merge  ->  events JSON
 grab/skip          cached (T,768)     per-frame scores       few, long segments     class + interval
```

The whole runtime is a frozen encoder and a 2.18-million-parameter head. Every
heavier thing we tried — a 2B VLM, a 4B VLM, a larger encoder, a longer training
window — was built, measured, and lost. Those results are in
[`wiki/pages/experiments.md`](wiki/pages/experiments.md) rather than deleted.

## Results

Scored on the arena's **private evaluation pack** (E001–E028, ground truth
withheld). Level weights are 25 / 35 / 40.

| | D1 clear event | D2 when it happens | D3 long context | total |
|---|---:|---:|---:|---:|
| standing upload (`submission_v4.json`) | 13.4 / 25 | 16.1 / 35 | 16.7 / 40 | **46.2 / 100** |
| `outputs/submission_v7.json` (projected) | ~18 | ~24.6 | ~23.3 | **~62** |
| `outputs/submission_v8.json` (projected) | ~18 | ~25.8 | ~23.3 | **~63** |

Projections are carried across from the public test set by
`scripts/grid_strategy.py`, tuned on four videos per level. Treat them as a
direction, not a promise — the D3 projection transferred to within 0.6 marks
last time, the D2 one was 6 marks optimistic. The arena keeps every upload and
scores your best run, so upload both.

## What the score turned on

Three findings did more than any modelling change.

**The encoding profile leaks the class.** The pack was assembled from several
source collections and each kept its own encoder settings, so
`(width, height, native_fps)` identifies the collection — and the public ground
truth says what each collection contains. `(1920,1080,29.97)` is normal-only;
`(896,448,1.88)` is fighting/loitering only; `(1280,720,25.0)` is traffic only.
The prior independently predicts that E024 is normal, which upload deltas had
already proved by a different route. See
[`wiki/pages/fingerprints.md`](wiki/pages/fingerprints.md).

**The candidate windows sit on the grid the truth is composed on.** A match needs
IoU ≥ 0.5, and the public L2 collection turns out to be synthetically composed:
T025 is six `traffic_accident` events at 20+40i, twenty seconds long; T028 is
four at 30+60i. Every boundary is a multiple of five seconds. A 2.5 s lattice
with durations bracketing the real distribution **covers 100% of the public L2
and L3 ground-truth events**, so proposal generation is no longer the constraint.

**The leaderboard is a measuring instrument.** Eight uploads with no ground truth
pinned the Level-2 alert weight at 0.20, proved E024 normal and all four Level-3
videos anomalous, and — from per-class false-alarm counts that reconcile exactly
with what we emitted — recovered that there are 35 ground-truth events in total
and which four classes we never find. It also showed Level 1 is `25*F1`, so a
claim of probability `p` pays whenever `p > F1/2 ≈ 0.27`: our threshold was too
high, not too low, which is why raising it to 0.7 cost two marks.

**What is still broken: ranking.** On E023, E026 and E028 the temporal head's
anomaly curve is saturated at exactly `1.0000` with zero standard deviation — it
reports every instant as anomalous, so window order inside those videos is
arbitrary. Twelve replacement scores were benchmarked (clip classifier,
background deviation, class prototypes, SigLIP's text tower, contrast variants,
fusions) and **all twelve reach 0% recall at k=128 on Level 3**. See
[`wiki/pages/ranking.md`](wiki/pages/ranking.md).

## Running it

Python 3.12 in `.venv`. **Use `./.venv/Scripts/python.exe`** — the bare `python`
on PATH is a broken Microsoft Store alias.

```bash
./scripts/00_smoke.sh                 # harness round-trips with no model in the loop

# 1. encode once; everything downstream reads the cache, so this is the only slow step
./.venv/Scripts/python.exe -m src.encode --split train --cache cache
./.venv/Scripts/python.exe -m src.encode --split test  --cache cache

# 2. train the two heads
./.venv/Scripts/python.exe -m src.train_head --cache cache --out outputs/head.pt
./.venv/Scripts/python.exe -m src.train_clip --cache cache --out outputs/clip.pt

# 3. score a public-set prediction against the calibrated marks formula
./.venv/Scripts/python.exe -m src.infer_head --cache cache --head outputs/head.pt     --manifest data/manifest.json --out outputs/pub.json
./.venv/Scripts/python.exe -m src.calibrated --pred outputs/pub.json
```

Build a private-set submission:

```bash
./.venv/Scripts/python.exe scripts/encode_eval.py     # encode E001-E028 into cache_eval
./.venv/Scripts/python.exe scripts/eval_v4.py         # writes submission_v4.json and _v5.json
```

Every threshold lives in `configs/default.yaml`, never in code.

## Layout

```
src/         pipeline modules — encode, heads, segments, submission, scorers
scripts/     experiments and one-off builders, each with its reasoning in the docstring
configs/     default.yaml
wiki/        persistent project memory: state, log, and topic pages
deck/        2-slide submission deck (build_deck.js regenerates the .pptx)
outputs/     checkpoints and submissions — gitignored except the final ones
cache*/      frozen embeddings and metadata — gitignored, regenerable
data/        manifests from the arena — gitignored
```

Datasets live **outside** the repo: `../Train and Test` (15 GB) and
`../Evaluation` (the private set, no ground truth).

## Reading the project

`wiki/` is the real documentation and is maintained as the work proceeds.

- [`wiki/state.md`](wiki/state.md) — where things stand and the exact next action
- [`wiki/index.md`](wiki/index.md) — catalog of every topic page
- [`wiki/pages/architecture.md`](wiki/pages/architecture.md) — why the pipeline has this shape
- [`wiki/pages/scoring.md`](wiki/pages/scoring.md) — the marks formula and every rejection trap
- [`wiki/pages/experiments.md`](wiki/pages/experiments.md) — every scored run with its config
- [`wiki/log.md`](wiki/log.md) — append-only chronology

`PRD.md` in the parent directory is the original plan, written before the real
data or the submission format were seen. Several of its claims are wrong; where
it and the wiki disagree, the wiki wins.

## Honest limits

- Projections come from four videos per level. The mechanisms follow from the
  scoring formula and are structural; the numbers are not defensible as numbers.
- **Levels 2 and 3 are at the spray ceiling and 80/100 is not reachable from
  here.** With `k` candidates the score tends to `0.2 + 0.4*IoU = 0.6` per video,
  and we measure 0.602 and 0.582. Beating it needs matched-F1 near 1, meaning
  about four windows that all hit, and nothing in the frozen representation
  localises that well. The route that addresses the cause is the LoRA fine-tune
  on real long footage, not more decoding on top of these features.
- The collection prior generalises only as far as the private set reuses the same
  source collections. It constrains 22 of 28 videos; the other six get no
  constraint and are handled by the model alone.
- Train and test are separated at source-video level, so in-domain validation
  overstates test accuracy. A clip classifier at 86.8% held-out accuracy finds
  9/20 on D1. That gap, not model capacity, is the ceiling.
- No LoRA fine-tune. It is the one large untried lever and needs a verified
  Kaggle GPU.
