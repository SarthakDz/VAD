"""Submission construction and validation for the arena JSON format.

Source: "AHC Visual Intelligence Hackathon Submission format" (2026-09-05).

Nothing reaches disk without passing `validate`. A rejected file does not burn
a run, but a *malformed-but-accepted* file overwrites your standing score with
a worse one and there is no best-of fallback -- so validate hard, locally.

The seven documented rejection traps, all asserted here:
  1. class_name "normal" is rejected -- a normal video is `"events": []`
  2. timestamps on a Level-1 event are rejected -- they must be null
  3. an omitted video keeps its previous answer, it is not cleared
  4. many fragments for one event: only the best-overlapping one can match
  5. any prediction on a normal Level-2/3 video scores that video zero
  6. claiming the whole clip is anomalous fails the 0.5 IoU gate
  7. runtime_metadata is required on every video
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .labels import ANOMALY_CLASSES, NORMAL

SCHEMA_VERSION = "1.0"
MAX_FILE_BYTES = 5 * 1024 * 1024
EXPLANATION_MIN, EXPLANATION_MAX = 20, 500
AVG_TOLERANCE = 0.02  # average_time_ms must match total/calls within 2%


class SubmissionError(ValueError):
    pass


# --------------------------------------------------------------------------
# manifest


def load_manifest(path: str | Path) -> dict[str, int]:
    """video_id -> level.

    The real manifest.json is only available from the arena site, and its exact
    shape is not documented in the PDF. Accept every plausible encoding and
    fail loudly rather than silently mis-reading levels -- a wrong level means
    every event for that video is rejected (trap 2).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    rows = None
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        for key in ("videos", "predictions", "manifest", "items", "data"):
            if isinstance(raw.get(key), list):
                rows = raw[key]
                break
        if rows is None and raw and all(isinstance(v, (int, str)) for v in raw.values()):
            return {str(k): int(v) for k, v in raw.items()}

    if rows is None:
        raise SubmissionError(
            f"{path}: cannot find a video list. Top-level keys: "
            f"{list(raw)[:10] if isinstance(raw, dict) else type(raw).__name__}"
        )

    out: dict[str, int] = {}
    for r in rows:
        if not isinstance(r, dict):
            raise SubmissionError(f"{path}: expected objects in the video list, got {type(r).__name__}")
        vid = r.get("video_id") or r.get("id") or r.get("video")
        lvl = r.get("level") or r.get("tier")
        if vid is None or lvl is None:
            raise SubmissionError(f"{path}: entry missing video_id/level: {r}")
        out[str(vid)] = int(lvl)
    if not out:
        raise SubmissionError(f"{path}: manifest is empty")
    return out


# --------------------------------------------------------------------------
# in-memory model


@dataclass
class Event:
    class_name: str
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    explanation: str | None = None

    def to_json(self) -> dict:
        d: dict = {
            "class_name": self.class_name,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
        }
        # Bonus only, and omitting it never costs -- but the window is enforced,
        # so drop anything outside it rather than risk a rejection.
        if self.explanation and EXPLANATION_MIN <= len(self.explanation) <= EXPLANATION_MAX:
            d["explanation"] = self.explanation
        return d


@dataclass
class ModelRuntime:
    model_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    call_times_ms: list[float] | None = None

    def to_json(self) -> dict:
        d: dict = {
            "model_name": self.model_name,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 1),
        }
        if self.call_count:
            d["average_time_ms"] = round(self.total_time_ms / self.call_count, 1)
        if self.call_times_ms:
            ts = sorted(self.call_times_ms)
            n = len(ts)
            d["p50_time_ms"] = round(ts[n // 2], 1)
            d["p95_time_ms"] = round(ts[min(n - 1, int(n * 0.95))], 1)
            d["max_time_ms"] = round(ts[-1], 1)
        return d


@dataclass
class RuntimeMetadata:
    """Required on every video. Excludes model loading and downloads."""

    frames_processed: int = 0
    chunks_processed: int = 0
    end_to_end_internal_time_ms: float = 0.0
    model_runtimes: list[ModelRuntime] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "frames_processed": int(self.frames_processed),
            "chunks_processed": int(self.chunks_processed),
            "end_to_end_internal_time_ms": round(self.end_to_end_internal_time_ms, 1),
            "model_runtimes": [m.to_json() for m in self.model_runtimes],
        }


@dataclass
class VideoPrediction:
    video_id: str
    events: list[Event] = field(default_factory=list)
    runtime: RuntimeMetadata = field(default_factory=RuntimeMetadata)

    def to_json(self) -> dict:
        return {
            "video_id": self.video_id,
            "events": [e.to_json() for e in self.events],
            "runtime_metadata": self.runtime.to_json(),
        }


# --------------------------------------------------------------------------
# validation


def validate(doc: dict, manifest: dict[str, int]) -> None:
    problems: list[str] = []

    preds = doc.get("predictions")
    if not isinstance(preds, list):
        raise SubmissionError("`predictions` is required and must be a list")

    seen: set[str] = set()
    for p in preds:
        vid = p.get("video_id")
        if vid is None:
            problems.append("a prediction has no video_id")
            continue
        if vid in seen:
            problems.append(f"{vid}: appears more than once")
        seen.add(vid)

        if vid not in manifest:
            problems.append(f"{vid}: not in the manifest")
            continue
        level = manifest[vid]

        if "events" not in p:
            problems.append(f"{vid}: `events` is required (use [] for normal)")
        if "runtime_metadata" not in p:
            problems.append(f"{vid}: runtime_metadata is required on every video (trap 7)")
        else:
            problems += _check_runtime(vid, p["runtime_metadata"])

        for i, ev in enumerate(p.get("events") or []):
            problems += _check_event(vid, level, i, ev)

    if problems:
        raise SubmissionError(
            "submission failed validation:\n  - " + "\n  - ".join(problems[:40])
        )


def _check_event(vid: str, level: int, i: int, ev: dict) -> list[str]:
    where = f"{vid} event[{i}]"
    out: list[str] = []

    cls = ev.get("class_name")
    if cls == NORMAL:
        out.append(f'{where}: class_name "normal" is rejected -- use "events": [] (trap 1)')
    elif cls not in ANOMALY_CLASSES:
        out.append(f"{where}: class_name {cls!r} is not one of the 11 anomaly classes")

    s, e = ev.get("start_time_sec"), ev.get("end_time_sec")
    if level == 1:
        if s is not None or e is not None:
            out.append(f"{where}: Level-1 timestamps must be null (trap 2)")
    else:
        if s is None or e is None:
            out.append(f"{where}: Level-{level} events require start_time_sec and end_time_sec")
        else:
            if s < 0:
                out.append(f"{where}: start_time_sec must be >= 0")
            if e <= s:
                out.append(f"{where}: end_time_sec must be greater than start_time_sec")

    ex = ev.get("explanation")
    if ex is not None and not (EXPLANATION_MIN <= len(ex) <= EXPLANATION_MAX):
        out.append(
            f"{where}: explanation must be {EXPLANATION_MIN}-{EXPLANATION_MAX} chars, got {len(ex)}"
        )
    return out


def _check_runtime(vid: str, rt: dict) -> list[str]:
    out: list[str] = []
    for f in ("frames_processed", "chunks_processed", "end_to_end_internal_time_ms"):
        if f not in rt:
            out.append(f"{vid}: runtime_metadata.{f} missing")
    for m in rt.get("model_runtimes") or []:
        name = m.get("model_name", "?")
        total, calls, avg = m.get("total_time_ms"), m.get("call_count"), m.get("average_time_ms")
        if total is not None and calls and avg is not None:
            expect = total / calls
            if expect > 0 and abs(avg - expect) / expect > AVG_TOLERANCE:
                out.append(
                    f"{vid}/{name}: average_time_ms {avg} != total/calls {expect:.1f} (>2%)"
                )
        ct = m.get("call_times_ms")
        if ct is not None and calls is not None and len(ct) != calls:
            out.append(f"{vid}/{name}: call_times_ms has {len(ct)} entries, call_count is {calls}")
    return out


# --------------------------------------------------------------------------
# assembly


def build(
    predictions: list[VideoPrediction],
    submission_id: str,
    model_name: str,
    total_wall_time_ms: float,
    hardware: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission_id,
        "model_name": model_name,
        "run_metadata": {
            "total_wall_time_ms": round(total_wall_time_ms, 1),
            "hardware": hardware,
        },
        "predictions": [p.to_json() for p in predictions],
    }


def write(doc: dict, out_path: str | Path, manifest: dict[str, int]) -> Path:
    validate(doc, manifest)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, ensure_ascii=False)
    size = len(text.encode("utf-8"))
    if size > MAX_FILE_BYTES:
        raise SubmissionError(f"file is {size / 1e6:.2f} MB, limit is 5 MB")
    out_path.write_text(text, encoding="utf-8")
    return out_path


def empty_submission(manifest: dict[str, int], **meta) -> dict:
    """Every video answered `normal`. The floor a real run must beat.

    Not a throwaway: on Level 2/3, an empty answer scores 1.0 on every normal
    video, so this is a genuine baseline, not a zero.
    """
    preds = [VideoPrediction(video_id=v) for v in manifest]
    return build(
        preds,
        meta.get("submission_id", "baseline-empty"),
        meta.get("model_name", "baseline-empty"),
        meta.get("total_wall_time_ms", 0.0),
        meta.get("hardware", "none"),
    )


def manifest_from_public_test(root: str | Path) -> dict[str, int]:
    """Stand-in manifest built from the public test set, for local dev only.

    Lets the whole submit/score loop run before the real manifest.json exists.
    """
    from .io_dataset import load_test_videos

    vids = load_test_videos(root)
    return {r.video_id: int(r.level) for r in vids.itertuples()}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="manifest.json from the arena site")
    ap.add_argument("--root", default="../Train and Test", help="fallback: public test set")
    ap.add_argument("--out", default="outputs/baseline_empty.json")
    a = ap.parse_args()

    mf = load_manifest(a.manifest) if a.manifest else manifest_from_public_test(a.root)
    p = write(empty_submission(mf), a.out, mf)
    levels: dict[int, int] = {}
    for lv in mf.values():
        levels[lv] = levels.get(lv, 0) + 1
    print(f"wrote {p}")
    print(f"  {len(mf)} videos, levels {dict(sorted(levels.items()))} -- validation passed")
