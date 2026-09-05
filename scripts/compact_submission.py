"""Shrink a submission's parameter count without changing its answers.

The arena rejected a 537-leaf submission with "Too many parameter values were
provided" -- a backend bind-parameter limit, not a schema problem. Our answers
are only 41 events across 34 videos; the bulk of the payload was 68
`model_runtimes` rows (2 per video x 4 fields = 272 leaves, over half the file).

Nothing here changes a single predicted class or timestamp, so the score is
unaffected. `end_to_end_internal_time_ms` is preserved in every variant, and
that is what the latency bonus is computed from -- the format PDF describes the
bonus as total reported processing time over total video duration, which does
not read `model_runtimes` at all.

Levels, smallest last:
  merged  one combined model_runtimes row per video instead of two
  bare    model_runtimes: []   (the format PDF's own T001 example does this)
  minimal bare, and videos with no events dropped entirely -- an omitted video
          is scored as normal, so this is identical in meaning for them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def leaves(o) -> int:
    if isinstance(o, dict):
        return sum(leaves(v) for v in o.values())
    if isinstance(o, list):
        return sum(leaves(v) for v in o)
    return 1


def merged(doc: dict) -> dict:
    out = json.loads(json.dumps(doc))
    for p in out["predictions"]:
        rt = p["runtime_metadata"]
        rows = rt.get("model_runtimes") or []
        if len(rows) > 1:
            total = sum(r.get("total_time_ms", 0.0) for r in rows)
            calls = sum(r.get("call_count", 0) for r in rows)
            rt["model_runtimes"] = [{
                "model_name": "siglip-gru-stage-a",
                "call_count": calls,
                "total_time_ms": round(total, 1),
                "average_time_ms": round(total / calls, 1) if calls else 0.0,
            }]
    return out


def bare(doc: dict) -> dict:
    out = json.loads(json.dumps(doc))
    for p in out["predictions"]:
        p["runtime_metadata"]["model_runtimes"] = []
    return out


def minimal(doc: dict) -> dict:
    out = bare(doc)
    out["predictions"] = [p for p in out["predictions"] if p["events"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/submission.json")
    ap.add_argument("--manifest", default="data/manifest.json")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.submit import load_manifest, validate

    mf = load_manifest(a.manifest)
    doc = json.loads(Path(a.src).read_text(encoding="utf-8"))

    variants = [("submission_merged.json", merged(doc)),
                ("submission_bare.json", bare(doc)),
                ("submission_minimal.json", minimal(doc))]

    print(f"{'file':32s}{'leaves':>8}{'videos':>8}{'events':>8}{'KB':>7}")
    print(f"{Path(a.src).name:32s}{leaves(doc):8d}"
          f"{len(doc['predictions']):8d}"
          f"{sum(len(p['events']) for p in doc['predictions']):8d}"
          f"{len(json.dumps(doc)) / 1024:7.1f}")
    for name, v in variants:
        validate(v, mf)  # every variant must still pass all 11 traps
        p = Path("outputs") / name
        p.write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{name:32s}{leaves(v):8d}{len(v['predictions']):8d}"
              f"{sum(len(x['events']) for x in v['predictions']):8d}"
              f"{len(json.dumps(v)) / 1024:7.1f}")
    print("\nall variants pass validation; answers are byte-identical in meaning")


if __name__ == "__main__":
    main()
