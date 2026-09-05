"""Stage C2: fuse the temporal head with the VLM.

Fusion policy, and the reason for it:

  * The head owns `start_time_sec` / `end_time_sec`. It is measurably good at
    boundaries -- on T028 it lands within 0.2 s of ground truth on all four
    events -- and the VLM sees only a handful of sampled frames, so it has
    strictly less information about *when* anything happened.
  * The VLM owns `class_name` and `explanation`. That is the head's measured
    weakness: confidently wrong labels on correctly located segments.
  * If the VLM declines to answer, or returns something outside the shortlist,
    the head's answer stands. Stage B may only improve on Stage A; a failed
    VLM call must never make the submission worse than the head alone.

There is no path here that suppresses a segment or invents one. Changing the
event count would put the fragmentation behaviour back in play, and that is
governed by `segments.py`, where it can be swept.
"""

from __future__ import annotations

import time

import numpy as np

from .segments import Segment
from .vlm import VLMStats, read_segment_frames, shortlist


def refine_segments(
    segments: list[Segment],
    video_path: str,
    class_prob: np.ndarray,
    vlm,
    stats: VLMStats,
    frames_per_segment: int = 6,
    use_motion_crop: bool = True,
    min_seconds: float = 0.0,
) -> tuple[list[Segment], dict[int, str]]:
    """Re-label each segment with the VLM. Timestamps are never touched.

    Returns the segments (classes possibly rewritten) and a map from segment
    index to the explanation the VLM produced, for the `explanation` field.
    """
    explanations: dict[int, str] = {}
    if not segments:
        return segments, explanations

    for i, seg in enumerate(segments):
        if (seg.end_sec - seg.start_sec) < min_seconds:
            continue

        frames = read_segment_frames(
            video_path, seg.start_sec, seg.end_sec,
            n=frames_per_segment, use_motion_crop=use_motion_crop,
        )
        if not frames:
            continue

        prob = class_prob[seg.lo:seg.hi].mean(axis=0) if seg.hi > seg.lo else None
        options = shortlist(seg.class_name, prob, k=6)

        t0 = time.perf_counter()
        try:
            ans = vlm.ask(frames, options, seg.end_sec - seg.start_sec)
        except Exception as e:  # a VLM failure must never sink the submission
            stats.record((time.perf_counter() - t0) * 1000.0, len(frames))
            stats.parse_failures += 1
            print(f"    VLM error on segment {i}: {type(e).__name__}: {e}")
            continue
        stats.record((time.perf_counter() - t0) * 1000.0, len(frames))

        if ans["class_name"]:
            seg.class_name = ans["class_name"]
        else:
            stats.parse_failures += 1
        if ans["explanation"]:
            explanations[i] = ans["explanation"]

    return segments, explanations
