"""Frame decoding and uniform temporal sampling.

Decode cost dominates M1, so the read loop matters. OpenCV's `read()` fully
decodes every frame; at FPS_SAMPLE=2 on 30fps footage we keep 1 frame in 15,
so 14 of every 15 full decodes would be thrown away. `grab()` advances the
demuxer without converting to a numpy array, so we grab-and-skip and only
`retrieve()` the frames we actually keep. On the 804 MB normal videos that is
the difference between minutes and tens of minutes.

Seeking with CAP_PROP_POS_FRAMES is not used: it is unreliable across the
codec mix in this pack (CCTV, dashcam and drone sources all differ) and can
silently land on the wrong frame.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMeta:
    video_id: str
    duration_sec: float
    native_fps: float
    native_frames: int
    width: int
    height: int
    sampled_frames: int
    sample_fps: float

    def to_json(self) -> dict:
        return asdict(self)


def probe(path: str | Path) -> tuple[float, float, int, int, int]:
    """(duration_sec, fps, n_frames, width, height) without decoding pixels."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    # Some files report nonsense; fall back to something sane rather than
    # dividing by zero and emitting NaN timestamps into a submission.
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    duration = n / fps if n > 0 else 0.0
    return duration, fps, n, w, h


def sample(
    path: str | Path,
    video_id: str,
    sample_fps: float = 2.0,
    max_frames: int = 4096,
    resize_to: int | None = None,
):
    """Yield (frames_rgb, meta). Frames are uniformly spaced at `sample_fps`.

    `max_frames` caps very long videos -- the effective sampling rate drops
    rather than the tail being truncated, so timestamps stay aligned to the
    whole video.
    """
    duration, fps, n_frames, w, h = probe(path)

    want = int(duration * sample_fps) if duration > 0 else max_frames
    want = max(1, min(want, max_frames))
    step = max(1, int(round(fps / sample_fps)))
    if n_frames > 0 and want > 0:
        step = max(1, n_frames // want)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")

    frames, idx = [], 0
    try:
        while len(frames) < want:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                if resize_to:
                    bgr = cv2.resize(bgr, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise OSError(f"decoded zero frames from {path}")

    # Trust the frames we actually got over the container's frame count.
    effective_fps = len(frames) / duration if duration > 0 else sample_fps
    meta = VideoMeta(
        video_id=video_id,
        duration_sec=round(duration, 3),
        native_fps=round(fps, 3),
        native_frames=n_frames,
        width=w,
        height=h,
        sampled_frames=len(frames),
        sample_fps=round(effective_fps, 4),
    )
    return np.stack(frames), meta


def timestamps(meta: VideoMeta) -> np.ndarray:
    """Wall-clock second of each sampled frame -- the bridge from head output
    back to start_time_sec / end_time_sec."""
    if meta.sampled_frames <= 1 or meta.duration_sec <= 0:
        return np.zeros(meta.sampled_frames, dtype=np.float32)
    return np.linspace(0.0, meta.duration_sec, meta.sampled_frames, dtype=np.float32)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--fps", type=float, default=2.0)
    a = ap.parse_args()

    t = time.time()
    fr, m = sample(a.path, Path(a.path).stem, a.fps)
    dt = time.time() - t
    print(m.to_json())
    print(f"shape {fr.shape}  decoded in {dt:.2f}s  "
          f"({m.duration_sec / dt:.1f}x realtime)")
