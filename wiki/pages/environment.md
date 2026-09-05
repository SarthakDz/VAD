# Environment

Windows 11. Shell is Git Bash (POSIX) with PowerShell also available — each takes
its own syntax.

## Paths

```
F:\flytbase\ahc-vad          this repo
F:\flytbase\Train and Test   the dataset, 15 GB, OUTSIDE the repo
F:\flytbase\*.pdf            the two organiser PDFs
F:\flytbase\hackathon.txt    extracted text of the main PDF
F:\flytbase\subfmt.txt       extracted text of the submission format PDF
```

Scripts default to `--root "../Train and Test"`.

## Python

**Always invoke `./.venv/Scripts/python.exe`.** The bare `python` on PATH is the
broken Microsoft Store alias and will fail with a store redirect message.

The venv is Python **3.12.0**. Python 3.14 is also installed system-wide and is
the `py` launcher default — **do not use it**, PyTorch has no wheels for 3.14.
That cost time to discover; it is why the venv exists.

Installed: `torch 2.6.0+cu124`, `torchvision`, `transformers 5.16.1`,
`opencv-python 5.0.0`, `pandas 3.0.5`, `numpy 2.5.2`, `tqdm`, `pyyaml`.

## GPU

```
NVIDIA GeForce RTX 4060 Laptop GPU
8.0 GB VRAM, driver 616.56, CUDA build 12.4
measured 5.4 TFLOPS fp32
```

8 GB is the binding constraint. It is fine for the frozen encoder and the 2.18M
temporal head, and fine for 4-bit VLM *inference*. It **cannot** LoRA a 4B VLM —
the vision tower activations on multi-image samples are what exhaust it, not the
weights.

## Compute split

Decided deliberately; see the reasoning preserved here so it is not re-litigated.

**Laptop handles M0, M1, M2, M3, M5** — harness, scorer, decode, encode, head
training, thresholds, benchmarking, demo. Plain PyTorch, `transformers`,
`opencv-python` and `av` all have working Windows wheels.

**Kaggle T4 x2 handles M4 only** — the LoRA fine-tune. Unsloth is Linux-only in
practice (triton), ms-swift assumes Linux, `flash-attn` has no Windows wheels,
`decord` does not build cleanly. That entire toolchain is the fine-tuning stack
and nothing else.

The trick that makes this cheap: **videos never cross the network.**

| | size |
|---|---|
| full video pack | 15–17 GB — never uploaded |
| SigLIP embeddings, all 3173 train videos | ~100 MB |
| SFT frames (~1500 samples x 8 JPEGs @ 512px) | ~700 MB |
| LoRA adapter coming back | ~100 MB |

Two Kaggle gotchas that eat afternoons: **internet is disabled by default** (toggle
*Internet on* in the sidebar or every pip install and HuggingFace download fails),
and **GPU stays locked until the account is phone-verified**.

WSL2 was considered and rejected — not installed, and setup plus a reboot costs
30–60 minutes for something only the fine-tune would need, and the fine-tune goes
to Kaggle anyway.

## Disk

100 GB free on F:, 373 GB on C:. Not a constraint.

## Git

`origin` = `https://github.com/SarthakDz/VAD.git`, branch `main`. Credential
helper is Git Credential Manager and push works without prompting.

`.gitignore` excludes `.venv/`, `cache/`, `outputs/`, `data/`, `__pycache__/`.
No weights, embeddings or dataset in the repo.

Git warns `LF will be replaced by CRLF` on every commit. Harmless, ignore it.

## Cache

```
cache/emb/{video_id}.npy      fp16 (T, 768) SigLIP embeddings
cache/meta/{video_id}.json    duration_sec, fps, decode_sec, encode_sec, ...
cache/scores/{video_id}.npy   fp16 (T, 13) anomaly + 12 class probabilities
```

All regenerable and gitignored. Full cache is 3207 files: 3173 train + 34 test.
Encoding the whole train split takes roughly 30 minutes at ~111 videos/min. Every
stage skips work already on disk, so a rerun is cheap.
