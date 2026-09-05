"""Prove every documented rejection trap actually fires."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.submit import SubmissionError, validate  # noqa: E402

MF = {"E001": 1, "E021": 2}


def rt(**kw):
    d = {"frames_processed": 10, "chunks_processed": 1,
         "end_to_end_internal_time_ms": 100.0, "model_runtimes": []}
    d.update(kw)
    return d


CASES = [
    ("trap 1  class_name normal",
     [{"video_id": "E001", "events": [{"class_name": "normal"}], "runtime_metadata": rt()}]),
    ("trap 2  L1 timestamps",
     [{"video_id": "E001", "runtime_metadata": rt(),
       "events": [{"class_name": "fire", "start_time_sec": 1, "end_time_sec": 2}]}]),
    ("trap 7  runtime_metadata missing",
     [{"video_id": "E001", "events": []}]),
    ("bad class string",
     [{"video_id": "E001", "events": [{"class_name": "Fire"}], "runtime_metadata": rt()}]),
    ("L2 event with null timestamps",
     [{"video_id": "E021", "runtime_metadata": rt(),
       "events": [{"class_name": "fire", "start_time_sec": None, "end_time_sec": None}]}]),
    ("end <= start",
     [{"video_id": "E021", "runtime_metadata": rt(),
       "events": [{"class_name": "fire", "start_time_sec": 5, "end_time_sec": 5}]}]),
    ("video not in manifest",
     [{"video_id": "E999", "events": [], "runtime_metadata": rt()}]),
    ("duplicate video_id",
     [{"video_id": "E001", "events": [], "runtime_metadata": rt()},
      {"video_id": "E001", "events": [], "runtime_metadata": rt()}]),
    ("average_time_ms off by >2%",
     [{"video_id": "E001", "events": [], "runtime_metadata": rt(model_runtimes=[
         {"model_name": "vlm", "call_count": 4, "total_time_ms": 400.0,
          "average_time_ms": 150.0}])}]),
    ("call_times_ms length mismatch",
     [{"video_id": "E001", "events": [], "runtime_metadata": rt(model_runtimes=[
         {"model_name": "vlm", "call_count": 4, "total_time_ms": 400.0,
          "average_time_ms": 100.0, "call_times_ms": [100.0, 100.0]}])}]),
    ("explanation too short",
     [{"video_id": "E001", "runtime_metadata": rt(),
       "events": [{"class_name": "fire", "start_time_sec": None,
                   "end_time_sec": None, "explanation": "fire"}]}]),
]

VALID = [
    {"video_id": "E001", "runtime_metadata": rt(),
     "events": [{"class_name": "fire", "start_time_sec": None, "end_time_sec": None,
                 "explanation": "Thick smoke rises from a burning structure."}]},
    {"video_id": "E021", "runtime_metadata": rt(model_runtimes=[
        {"model_name": "vlm", "call_count": 4, "total_time_ms": 400.0,
         "average_time_ms": 100.0, "call_times_ms": [90.0, 100.0, 105.0, 105.0]}]),
     "events": [{"class_name": "traffic_accident", "start_time_sec": 20, "end_time_sec": 40}]},
]

fails = 0
for name, preds in CASES:
    try:
        validate({"predictions": preds}, MF)
        print(f"  NOT CAUGHT  {name}")
        fails += 1
    except SubmissionError:
        print(f"  caught      {name}")

try:
    validate({"predictions": VALID}, MF)
    print("  accepted    valid submission")
except SubmissionError as e:
    print(f"  FALSE REJECT of valid submission: {e}")
    fails += 1

print("\nFAIL" if fails else "\nALL VALIDATION CHECKS PASS")
sys.exit(1 if fails else 0)
