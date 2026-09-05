# CLAUDE.md — read this first, every session

This repo is an entry in the **AHC Visual Intelligence Hackathon** (near-real-time
video anomaly detection). It carries a maintained wiki in `wiki/` that acts as
persistent memory across sessions. You did not write it in this session; a
previous session did. Trust it, then verify anything it says about code.

## Start-of-session protocol

Do these three things before answering anything, in order:

1. **Read `wiki/state.md`.** It is the single source of truth for where the work
   stands: current milestone, what just finished, what is running, the exact next
   action, and what is blocked on the user. It is short by design.
2. **Read `wiki/index.md`.** It catalogs every wiki page with a one-line summary.
   Use it to find the pages relevant to whatever the user just asked, then read
   those. Do not read every page by default.
3. **Verify anything time-sensitive** before relying on it — a background job the
   state file claims is running, a file count, a cached artifact. State files go
   stale; the filesystem does not.

Then answer. Do not re-derive facts the wiki already records, and do not re-audit
the dataset — `wiki/pages/dataset-audit.md` holds the findings and how they were
obtained.

## End-of-session protocol

Whenever you finish a meaningful unit of work — a milestone, an experiment with a
score, a discovered constraint, a decision the user made — update the wiki before
you finish your reply:

- **`wiki/state.md`** — rewrite the current-position section. This file is
  overwritten, never appended to. Keep it under ~60 lines.
- **`wiki/log.md`** — append one entry. Format is fixed so it stays greppable:
  `## [YYYY-MM-DD HH:MM] <kind> | <one-line title>` where `<kind>` is one of
  `milestone`, `experiment`, `decision`, `finding`, `blocker`, `session`.
- **The relevant topic page** in `wiki/pages/` — fold the new fact in. Do not
  create a new page for something an existing page covers.
- **`wiki/index.md`** — only if you added a page.

Every experiment that produces a score goes in `wiki/pages/experiments.md` with
its config, or the numbers are lost and get re-run.

## Wiki conventions

- Wiki prose is normal English, written for a future reader with no context.
  This holds even when the user has asked for terse output in chat — chat style
  and file style are separate.
- Cross-link with `[[page-name]]`, matching the filename without `.md`.
- Record **why**, not just what. A threshold value is nearly useless; a threshold
  value plus the failure it prevents is durable.
- When a new source contradicts a wiki claim, fix the claim and note the
  correction in `log.md`. Do not leave both versions standing.
- Numbers get their provenance: measured, assumed, or quoted from a document.
  Assumptions must say so — several scoring weights are currently guesses.

## Repo layout

```
src/          pipeline modules (see wiki/pages/architecture.md)
scripts/      smoke tests and sanity checks
configs/      default.yaml — every threshold lives here, never in code
wiki/         persistent memory (this system)
cache/        embeddings, meta, score curves — gitignored, regenerable
outputs/      checkpoints and submissions — gitignored
data/         manifest.json etc. from the arena — gitignored
```

The dataset itself is **outside** the repo at `../Train and Test`.

## Hard rules for this project

- **Never put a hosted model in the runtime path.** Explicitly forbidden by the
  organisers. Large models are for development, comparison and generating
  training data only.
- **Never write a submission without `src.submit.validate` passing.** The arena
  has no best-of: a worse upload permanently replaces a better score.
- **Never drop a video.** A failure emits an empty answer and logs it.
- Run Python through `./.venv/Scripts/python.exe`. The bare `python` on PATH is a
  broken Microsoft Store alias.
