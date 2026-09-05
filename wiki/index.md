# Index

Catalog of the wiki. Read [[state]] first for where the work stands, then come
here to find the pages relevant to the question at hand. Do not read every page
by default.

## Navigation

| page | what it holds |
|---|---|
| [[state]] | **Read first.** Current milestone, what is running, exact next action, what is blocked. Overwritten each session, never appended to. |
| [[log]] | Append-only chronological record. `grep "^## \[" wiki/log.md \| tail -5` for recent activity. |

## Topics

| page | what it holds |
|---|---|
| [[dataset-audit]] | Everything known about the 15 GB pack: the two-schema difference, class distribution, why supervision is dense but localisation signal is thin, the train/test structural gap, templated descriptions, label noise, public test profile. |
| [[scoring]] | The arena JSON format, the eleven valid class strings, all rejection traps, the scoring formula, which weights are assumed rather than known, and verified scorer behaviour including the fragmented-oracle result. |
| [[architecture]] | The three-stage pipeline, what each `src/` module does and why, the decode and encode tricks, how long training sequences are synthesised, and the segment logic that decides the Level 2/3 score. |
| [[environment]] | Paths, the Python 3.12 venv, GPU limits, the laptop-versus-Kaggle compute split and why videos never cross the network, cache layout, git remote. |
| [[milestones]] | M0 through M5 with acceptance criteria and status, final deliverables, and an explicit list of things not to do. |
| [[fingerprints]] | The encoding profile `(width, height, fps)` identifies the source collection, what each collection contains, why the prior is trustworthy, and how it constrains classes on D1 and the D2/D3 spray. |
| [[ranking]] | Why Levels 2 and 3 are capped: the lattice covers every truth, the head's anomaly curve is saturated, and twelve replacement rankers benchmarked. |
| [[d1]] | The Level-1 classifier: the F1 break-even rule for claiming, the retrieval and text-tower members, and what still fails. |
| [[experiments]] | Every scored run with its full config. Reference points at the top. Newest last. |
| [[prior-art]] | What the organisers's SOTA deck and the AI City Challenge paper contain: the VAD/VAU/VAR ladder, Track 3 leaderboard, and the named methods worth borrowing from. |
| [[open-questions]] | Questions for the organisers in paste-ready prose, and manual tasks only the user can do. |

## Where to look for a given question

- *"What should I do next?"* → [[state]]
- *"Why is Level 2 scoring badly?"* → [[state]] diagnosis, then [[scoring]] and
  [[architecture]]'s segment section
- *"Can I trust this field / label / column?"* → [[dataset-audit]]
- *"Will this submission be rejected?"* → [[scoring]] traps, then run
  `scripts/test_validation.py`
- *"What did we already try?"* → [[experiments]]
- *"Why was it built this way?"* → [[architecture]]
- *"Why can't I just run this on the laptop?"* → [[environment]]
- *"Has someone solved this already?"* → [[prior-art]]
- *"What are we waiting on?"* → [[open-questions]]

## Source documents

Outside the repo, in `F:\flytbase\`:

- `AHC Visual Intelligence Hackathon.pdf` — problem statement, dataset doc,
  prerequisites, primer. Extracted text at `hackathon.txt`.
- `AHC Visual Intelligence Hackathon Submission format.pdf` — the arena format,
  traps and scoring. Extracted text at `subfmt.txt`.
- `PRD.md` — the original plan, written before either the real data or the
  submission format was seen. **Several of its claims are wrong**; where it and
  this wiki disagree, the wiki wins. Specifically: it assumed a shared train/test
  schema, assumed a CSV submission, assumed blank descriptions, and planned MIL
  pooling that the data makes unnecessary.
