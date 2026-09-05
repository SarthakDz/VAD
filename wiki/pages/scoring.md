# Scoring and submission format

Source: *AHC Visual Intelligence Hackathon Submission format* PDF, read
2026-09-05. Implemented in `src/submit.py` (writing, validation) and
`src/score.py` (local scoring). Verified by `scripts/test_validation.py` and
`scripts/sanity_score.py`.

## The submission is JSON, not CSV

An earlier version of this project wrote CSV. That was wrong and has been
replaced. Shape:

```json
{
  "schema_version": "1.0",
  "submission_id": "my-team-run-01",
  "model_name": "my-near-real-time-vad",
  "run_metadata": { "total_wall_time_ms": 1476000, "hardware": "1x RTX 4090" },
  "predictions": [
    {
      "video_id": "E001",
      "events": [],
      "runtime_metadata": {
        "frames_processed": 90,
        "chunks_processed": 1,
        "end_to_end_internal_time_ms": 4820,
        "model_runtimes": []
      }
    }
  ]
}
```

Only `predictions` is strictly required at top level. Per video, all three of
`video_id`, `events` and `runtime_metadata` are required.

## The eleven valid class strings

`class_name` accepts only the anomaly classes. **`normal` is not one of them** —
a normal video is expressed as `"events": []`.

```
traffic_accident   traffic_congestion   stalled_or_broken_down_vehicle
vehicle_blocking_traffic   wrong_way_driving   road_spill_or_debris
waterlogging_or_flood   fire   smoke   fighting_or_violence
loitering_or_suspicious_presence
```

## Rejection traps

All eleven of these are asserted in `src/submit.py::validate` and covered by
`scripts/test_validation.py`, which currently passes with no false rejects.

1. `"class_name": "normal"` is rejected — use `"events": []`
2. Timestamps on a Level-1 event are rejected — they must be `null`
3. An omitted video keeps its previous answer; it is **not** cleared
4. Many fragments for one event: only the best-overlapping one can match
5. Any prediction on a normal Level-2/3 video scores that video zero
6. Claiming the whole clip is anomalous fails the 0.5 IoU gate
7. `runtime_metadata` is required on every video

Plus, enforced by the same validator: `end_time_sec` must exceed
`start_time_sec`, `start_time_sec` must be ≥ 0, `video_id` must appear exactly
once and be in the manifest, `explanation` must be 20–500 characters,
`average_time_ms` must match `total_time_ms / call_count` within 2%, and
`call_times_ms` must have exactly `call_count` entries. Max file size 5 MB.

Unknown extra fields (confidence, bbox, debug keys) are ignored, not rejected.

## Scoring scheme

```
Level 1    pooled over all Level-1 videos:
               0.5 * (anomaly vs normal accuracy) + 0.5 * (class accuracy)
Level 2/3  scored per video, then averaged:
               ground truth normal, predicted nothing   -> 1
               ground truth normal, predicted anything  -> 0
               ground truth has events -> weighted mix of alert / matched / timing,
                                          timing weighted higher at Level 3
```

An event matches only when **the class is right AND IoU ≥ 0.5**. One predicted
event can match at most one ground-truth event; the rest count against you.

The PDF phrases the gate usefully: *if your interval sits inside the real event
it must cover at least half of it; if it swallows the real event it must be no
more than twice as long.*

**Latency bonus:** total reported processing time ÷ total video duration.

## ASSUMPTION — the component weights are guessed

The exact alert / matched / timing weights are **not published**. `src/score.py`
currently uses:

```
Level 2   (0.2, 0.5, 0.3)
Level 3   (0.2, 0.4, 0.4)     timing heavier, per the PDF's wording
```

These are configurable via `--w2` / `--w3` and mirrored in
`configs/default.yaml`. **Ask the organisers and replace them.** Components are
always printed separately so a weight change never hides a weak component.

## Verified scorer behaviour

`scripts/sanity_score.py` against the public test set:

| strategy | L1 | L2 | L3 | overall |
|---|---:|---:|---:|---:|
| oracle (perfect) | 1.000 | 1.000 | 1.000 | **1.000** |
| empty (all normal) | 0.167 | 0.333 | 0.000 | 0.167 |
| whole-clip accident | 0.479 | 0.133 | 0.200 | 0.271 |
| spam all classes | 0.479 | 0.133 | 0.200 | 0.271 |
| **fragmented oracle** | 1.000 | **0.467** | **0.200** | 0.556 |

The oracle scoring exactly 1.000 is the proof the arithmetic is right.

**The fragmented-oracle row is the most important number in this project.**
Perfect events with correct classes, merely shattered into five slices each,
collapse Level 2 from 1.000 to 0.467 and Level 3 from 1.000 to 0.200. Segment
merging matters more than detection accuracy at Levels 2 and 3. This is what
drives the conservative defaults in [[architecture]].

## Two non-obvious strategic consequences

**The empty submission is not zero — it scores 0.333 at Level 2.** Two of six
Level-2 videos are normal, and an empty answer scores 1.0 on each. Since an
unanswered video is scored as normal, **omitting a low-confidence video is a
real strategy, not a forfeit.**

**There is no best-of.** Every upload re-scores everything and overwrites the
previous score; a worse run permanently replaces a better one. Never upload
without the local scorer passing first. Rejected files do not consume a run, but
accepted-and-worse ones cost you your standing.

## Submission mechanics

- Sign in to the arena site with the registered Google account. **URL is not in
  any provided PDF** — see [[open-questions]].
- Download `manifest.json` (the video list and each one's level) and the starter
  template from the Benchmark tab.
- A file only updates the videos it mentions; other answers persist.
- Each upload costs one run regardless of how many videos it covers.
- Final submission section requires: code repository URL, architecture write-up
  (link or PDF/HTML up to 25 MB), and a **2-slide PPT stated to carry high
  weightage**.
