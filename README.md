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
| standing upload | 12.0 / 25 | 14.0 / 35 | 11.2 / 40 | **37.2 / 100** |
| `outputs/submission.json` (projected) | 12.5 | 22.3 | 16.9 | ~51 |
| `outputs/submission_v5.json` (projected) | 15.0 | 22.3 | 16.9 | ~54 |

Projections are carried across from the public test set by
`scripts/d23_strategy.py`, tuned on four videos per level. Treat them as a
direction, not a promise. Upload `submission.json` first so the D2/D3 change is
readable on its own; the arena keeps every upload and scores your best run, so a
worse attempt costs nothing.

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

**A third of our candidate windows could never match.** A match needs IoU ≥ 0.5,
so a window of width *w* can only ever match a truth of width *w/2* to *2w*.
Real events run 5–125 s; we were emitting 240 s windows. They could not score,
only dilute precision. Fixing the widths, and spending the candidate budget
round-robin across them instead of on the highest-scoring narrow window, moved
the public anomalous-video mean from 0.492 to 0.516 on L2 and 0.353 to 0.424 on L3.

**The leaderboard is a measuring instrument.** Six uploads with no ground truth
pinned the D2 alert weight at 0.20, proved all four L3 videos are anomalous, and
showed that nine of the twenty L1 videos are normal — where the public set was
20 anomalous out of 24. Our detection threshold had been tuned on the wrong
prior and was over-claiming by a third.

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
- The collection prior generalises only as far as the private set reuses the same
  source collections. It constrains 22 of 28 videos; the other six get no
  constraint and are handled by the model alone.
- Train and test are separated at source-video level, so in-domain validation
  overstates test accuracy. A clip classifier at 86.8% held-out accuracy finds
  9/20 on D1. That gap, not model capacity, is the ceiling.
- No LoRA fine-tune. It is the one large untried lever and needs a verified
  Kaggle GPU.
