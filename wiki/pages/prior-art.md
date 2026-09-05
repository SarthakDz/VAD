# Prior art

From two documents the organisers shared on 2026-09-05: their state-of-the-art
deck (`AHC_VAD_HACKATHON_SOTA.pptx.pdf`, extracted to `sota.txt`) and
`research paper.pdf`, which is **The 10th AI City Challenge**, arXiv:2608.17044.

The research paper is a scanned PDF with no text layer. It was read by rendering
pages to PNG with PyMuPDF into `F:\flytbase\paper_pages\`; `pdftotext` returns
nothing on it.

## The organisers' framing: the problem ladder

The deck frames the field as renaming the problem twice in two years, each rung
adding an output the one below cannot produce:

- **VAD** (2018–2024) — Video Anomaly Detection. A score per frame. Is something
  wrong, and when? No account of *what*.
- **VAU** (2024–2025) — Video Anomaly Understanding. Detection plus description
  and explanation.
- **VAR** (2025–2026) — Video Anomaly Reasoning. Explicit multi-stage reasoning
  before the answer: perception, then cognition, then verdict.

Our task spans VAD (the Level 2/3 timing) and VAU (the `explanation` field). Good
framing for the 2-slide deck, since it positions the cascade rather than
apologising for it.

## AI City Challenge Track 3 — the directly comparable benchmark

Track 3 introduces **TAR** (Traffic Anomaly Reasoning) and **TAR-Bench**: 44,040
annotations over 3,670 transportation videos across 10 task types, with 960
human-curated annotations over 80 held-out clips.

**Leaderboard, accepted papers only (Table 3):**

| rank | team | mean | base model |
|---|---|---:|---|
| 1/27 | Stellarview AI | 0.6788 | Qwen 3.5 |
| – (2/76) | UOB&UW Team | 0.6779 | Qwen3VL-8B |
| 2/27 | FPT AI Vision | 0.6703 | Qwen3-VL-8B |
| 3/27 | Smart Vision | 0.6669 | Qwen |
| 10/27 | UWIPL_ETRI | 0.6185 | Qwen |
| 15/27 | OptimAI | 0.5880 | Qwen |
| 16/27 | MR-CAS | 0.5780 | GPT |
| 24/27 | Korea Drive | 0.4256 | Qwen3 |

Three things worth carrying forward.

**Qwen dominates and a GPT-based entry placed 16th.** The Qwen3-VL family is the
right bet; our M4 plan targets Qwen3-VL-4B, one size below the 8B that took 2nd
and 3rd. Nothing here suggests a hosted frontier model would have helped.

**The ceiling is 0.6788.** With 325 registered teams and a full challenge cycle,
nobody got near 1.0. Useful calibration for what a good score looks like.

**The paper's own conclusion validates the cascade.** Quoting the Track 3
summary: *"The accepted papers indicate a shift from simple VLM prompting toward
agentic pipelines that first extract visual evidence, then match it to a
task-specific answer format."* That is exactly Stage A extracting candidate
segments and Stage B formatting them. Worth citing on the slides.

## The most important difference from our task

From §4.3 Track 3 Evaluation:

> *"The official in-domain TAR-Bench mean was computed from nine scored task
> types, **excluding temporal localization**. Closed-form tasks used
> accuracy-style scoring, while text-generation tasks used BLEU, METEOR, ROUGE,
> and CIDEr."*

**AI City Track 3 does not score temporal localisation. Our hackathon does, with a
hard IoU 0.5 gate and timing weighted higher at Level 3.**

Two consequences. Their leaderboard numbers are not comparable to ours. And the
published state-of-the-art methods were not optimised for the dimension we are
scored hardest on — which is the same conclusion [[scoring]] reached from the
fragmented-oracle result, arrived at independently. Temporal localisation is
genuinely the differentiator here.

## Methods named in the deck

**Cosmos-Embed1-448p-anomaly-detection** (NVIDIA). Detection by *retrieval*
rather than classification. A five-second clip goes through a video tower into a
768-dim vector; the sentence "wrong-way driving" goes through a text tower into
the same space. There is no classifier head — the LoRA fine-tune only pulled clip
vectors closer to the phrases describing them. Score every clip by cosine
similarity to a query and sort; the top-k are the detections and the query text
*is* the class list. Evaluated on a 438-video Vad-Reasoning test split.

Confirmed live on HuggingFace: `nvidia/Cosmos-Embed1-448p-anomaly-detection`,
25 files, ~6.6k downloads, pipeline `video-classification`, **license "other"** —
which needs checking against open question 13 in [[open-questions]] before it
goes anywhere near the runtime path.

Why it matters here: it is purpose-built for exactly our task and outputs the
same 768 dimensions as `siglip-base-patch16-224`. The caveat is that it encodes
a *clip* to one vector, not a frame, so it is not a literal drop-in for our
per-frame Stage A — swapping it changes temporal resolution and would need the
head's timestep semantics rethought. Attractive but not free. See
[[architecture]].

**SlowFastVAD** (Ding et al., arXiv:2504.10320). A fast detector and a
RAG-enhanced VLM, fused. Structurally the same idea as our Stage A / Stage B
cascade; useful as a citation for why the design is sound.

**EventVAD** (Shao et al., arXiv:2504.13092, ACM MM 2025,
github.com/YihuaJerry/EventVAD). Training-free: cut at event boundaries, then let
an MLLM score each event. **Code is released**, unlike Cerberus. Directly
relevant to `src/segments.py`, which is doing event-boundary cutting — worth a
look if the threshold sweep stalls.

**QVAD** (Hamza, Karim, Nguyen, Yilmaz, arXiv:2604.03040). Thesis: *"the
bottleneck is not model capacity, it is the static question."* Argues for adaptive
querying rather than one fixed prompt. Relevant to Stage B prompt design and
aligns with the shortlist-prompting plan in [[architecture]].

**TAU-Agent** (Lin et al., arXiv:2608.25935, github.com/siri-rouser/TAU-Agent).
Agentic retrieval-augmented traffic anomaly understanding, from AI City 2026.

**GridVAD** — named in the deck with no detail given.

## Other context from the deck

Block 01 covers physical AI and world-action models: GPT-6 Astra, robotics'
"GPT-3 moment", DreamZero scoring 1750 against Pi-0.5's 1622 on the April 2026
RoboArena leaderboard trained only on DROID, and Qwen3.6. Background framing for
the session, not actionable for our build.
