"""Stage B: a small VLM that arbitrates the class of a candidate segment.

This targets one specific, measured failure. On T025 the temporal head finds
all six accident intervals almost exactly and labels every one
`wrong_way_driving` with 98.2% of the class mass -- `traffic_accident` is not
even in its top four. Six of the sixteen Level-2 ground-truth events are lost
to that single error, and no threshold or vote change can fix a confidently
wrong classifier. Boundaries are already right; only the label is wrong.

So Stage B deliberately does NOT touch timestamps. The head owns `when`, the
VLM owns `what`. That split also keeps the cost story intact: the VLM sees a
handful of frames per candidate segment rather than the whole video.

Two things the audit and the prior art both point at:

  * Ask for a shortlist, not 12 flat options. `labels.CONFUSABLE_GROUPS`
    encodes the confusions that actually dominate (fire/smoke,
    congestion/blocking/stalled, normal/loitering, accident/stalled).
  * Crop to the motion region. Frames are 224px by the time they reach the
    encoder, and small distant objects in 1080p drone footage are gone. The
    VLM reads ORIGINAL frames, and frame-differencing the segment finds where
    to look -- the Cerberus motion-mask idea, roughly 20 lines, no released
    code.

Runtime is measured per call so `runtime_metadata.model_runtimes` reports what
actually happened rather than an estimate.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .labels import ANOMALY_CLASSES, CONFUSABLE_GROUPS, NORMAL

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

PROMPT = """You are reviewing drone, CCTV or dashcam footage for a traffic incident operator.

These {n} frames are consecutive samples from a {dur:.0f}-second window that an \
earlier detector flagged as anomalous. Decide which single category best \
describes what is happening in this window.

Choose exactly one of:
{options}

Answer with JSON only, no other text:
{{"class_name": "<one of the options above>", "explanation": "<one sentence, 20 to 200 characters, describing what you actually see>"}}"""


@dataclass
class VLMStats:
    call_count: int = 0
    total_time_ms: float = 0.0
    call_times_ms: list[float] = field(default_factory=list)
    frames_seen: int = 0
    parse_failures: int = 0

    def record(self, ms: float, n_frames: int) -> None:
        self.call_count += 1
        self.total_time_ms += ms
        self.call_times_ms.append(ms)
        self.frames_seen += n_frames


# --------------------------------------------------------------------------
# frame selection


def motion_crop(frames: np.ndarray, pad: float = 0.15,
                min_frac: float = 0.20) -> tuple[int, int, int, int] | None:
    """Bounding box of where the scene is changing, as (x0, y0, x1, y1).

    Frame-difference the stack, blur, threshold at a high percentile, and take
    the extent of what survives. Returns None when motion is diffuse -- a
    moving drone camera makes the whole frame "move", and cropping to that is
    both pointless and risks discarding the subject.
    """
    if len(frames) < 2:
        return None
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]).astype(np.float32)
    diff = np.abs(np.diff(g, axis=0)).mean(axis=0)
    diff = cv2.GaussianBlur(diff, (0, 0), 3)
    if diff.max() <= 1e-6:
        return None
    thr = np.percentile(diff, 97.0)
    mask = diff >= max(thr, diff.max() * 0.25)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    h, w = diff.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x1 - x0) * (y1 - y0) > 0.85 * h * w:
        return None  # diffuse motion, likely camera ego-motion

    px, py = int((x1 - x0) * pad) + 8, int((y1 - y0) * pad) + 8
    x0, x1 = max(0, x0 - px), min(w, x1 + px)
    y0, y1 = max(0, y0 - py), min(h, y1 + py)
    # Never hand over a sliver: keep at least min_frac of each axis, centred.
    if (x1 - x0) < w * min_frac:
        c = (x0 + x1) // 2
        half = int(w * min_frac / 2)
        x0, x1 = max(0, c - half), min(w, c + half)
    if (y1 - y0) < h * min_frac:
        c = (y0 + y1) // 2
        half = int(h * min_frac / 2)
        y0, y1 = max(0, c - half), min(h, c + half)
    return int(x0), int(y0), int(x1), int(y1)


def read_segment_frames(path: str, start_sec: float, end_sec: float, n: int = 6,
                        max_side: int = 640, use_motion_crop: bool = True):
    """`n` evenly spaced ORIGINAL-resolution frames from [start_sec, end_sec]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1.0 <= fps <= 240.0):
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        lo, hi = int(start_sec * fps), int(end_sec * fps)
        if total:
            lo, hi = max(0, min(lo, total - 1)), max(1, min(hi, total))
        if hi <= lo:
            hi = lo + 1
        want = np.linspace(lo, max(lo, hi - 1), n).astype(int)

        out, idx, wi = [], 0, 0
        while wi < len(want):
            if not cap.grab():
                break
            if idx == want[wi]:
                ok, bgr = cap.retrieve()
                if ok:
                    out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                while wi < len(want) and want[wi] == idx:
                    wi += 1
            idx += 1
    finally:
        cap.release()

    if not out:
        return []
    arr = np.stack(out)
    if use_motion_crop:
        box = motion_crop(arr)
        if box:
            x0, y0, x1, y1 = box
            arr = arr[:, y0:y1, x0:x1]
    # Downscale only; the point is to keep detail the 224px encoder threw away.
    h, w = arr.shape[1:3]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        arr = np.stack([cv2.resize(f, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA) for f in arr])
    return list(arr)


# --------------------------------------------------------------------------
# shortlist


def shortlist(head_class: str, class_prob: np.ndarray | None = None,
              k: int = 5) -> list[str]:
    """Candidate classes for one segment: the head's pick, its confusable
    group, then the next most probable classes until we have `k`."""
    out = [head_class]
    for c in _group(head_class):
        if c not in out:
            out.append(c)
    if class_prob is not None:
        from .labels import CLASSES
        for i in np.argsort(-class_prob):
            c = CLASSES[i]
            if c != NORMAL and c not in out:
                out.append(c)
            if len(out) >= k:
                break
    return [c for c in out if c in ANOMALY_CLASSES][:k]


def _group(cls: str) -> list[str]:
    out: list[str] = []
    for g in CONFUSABLE_GROUPS:
        if cls in g:
            out += [c for c in g if c != cls]
    return out


# --------------------------------------------------------------------------
# model


class QwenVLM:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda",
                 dtype: str = "bfloat16", max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=device,
        ).eval()
        self.name = model_name

    def ask(self, frames: list[np.ndarray], options: list[str], duration: float) -> dict:
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames]
        text = PROMPT.format(
            n=len(imgs), dur=duration,
            options="\n".join(f"  - {o}" for o in options),
        )
        messages = [{"role": "user", "content":
                     [{"type": "image", "image": im} for im in imgs]
                     + [{"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return parse_answer(self.processor.decode(gen, skip_special_tokens=True), options)


def parse_answer(text: str, options: list[str]) -> dict:
    """Pull {class_name, explanation} out of a model reply.

    Small models wander off JSON, so fall back to substring matching before
    giving up. Returning None for class_name means "keep the head's answer",
    which is the safe default -- Stage B may only improve on Stage A.
    """
    cls, expl = None, None
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            c = str(d.get("class_name", "")).strip()
            if c in options:
                cls = c
            e = d.get("explanation")
            if isinstance(e, str) and e.strip():
                expl = e.strip()
        except Exception:
            pass
    if cls is None:
        hits = [o for o in options if o in text]
        if len(hits) == 1:
            cls = hits[0]
    return {"class_name": cls, "explanation": expl, "raw": text}
