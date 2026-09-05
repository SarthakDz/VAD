"""Frames -> frozen SigLIP embeddings -> cache/emb/{video_id}.npy

The encoder is never fine-tuned. Everything downstream (the temporal head, the
segment logic) reads these cached arrays, so this runs once and every later
experiment is cheap. A rerun skips whatever is already on disk -- the single
highest-leverage engineering decision in the project.

Also writes cache/meta/{video_id}.json, which carries duration_sec. That is
needed twice: to convert head timesteps back into wall-clock timestamps, and
to compute the arena's latency bonus (reported time / video duration).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .frames import sample


def load_encoder(name: str, device: str, dtype: torch.dtype):
    # AutoImageProcessor, not AutoProcessor: the latter also builds the text
    # tokenizer, which pulls in SentencePiece for no benefit -- we only ever
    # run the vision tower.
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    vision = getattr(model, "vision_model", model)
    return proc, vision


@torch.inference_mode()
def encode_frames(frames: np.ndarray, proc, vision, device: str,
                  dtype: torch.dtype, batch_size: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(frames), batch_size):
        chunk = list(frames[i:i + batch_size])
        px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(device, dtype)
        feats = vision(pixel_values=px).pooler_output
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def run(
    items: list[tuple[str, str]],
    cache_dir: Path,
    model_name: str,
    sample_fps: float,
    batch_size: int,
    max_frames: int,
    device: str,
    resize: int = 224,
    overwrite: bool = False,
) -> dict:
    emb_dir, meta_dir = cache_dir / "emb", cache_dir / "meta"
    emb_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    todo = [(v, p) for v, p in items
            if overwrite or not (emb_dir / f"{v}.npy").exists()]
    print(f"{len(items)} videos, {len(items) - len(todo)} cached, {len(todo)} to encode")
    if not todo:
        return {"encoded": 0, "skipped": len(items), "failed": 0}

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {model_name} on {device} ...")
    proc, vision = load_encoder(model_name, device, dtype)

    # Longest first: the 800 MB normal videos dominate wall-clock, and hitting
    # them early surfaces OOM or codec problems before the cheap 90% has run.
    todo.sort(key=lambda vp: Path(vp[1]).stat().st_size, reverse=True)

    ok = fail = 0
    t_start = time.time()
    total_secs = total_frames = 0.0

    for i, (vid, path) in enumerate(todo, 1):
        try:
            t0 = time.time()
            frames, meta = sample(path, vid, sample_fps, max_frames, resize_to=resize)
            t_dec = time.time() - t0

            t1 = time.time()
            emb = encode_frames(frames, proc, vision, device, dtype, batch_size)
            t_enc = time.time() - t1

            np.save(emb_dir / f"{vid}.npy", emb)
            m = meta.to_json()
            m.update(embed_dim=int(emb.shape[1]),
                     decode_sec=round(t_dec, 3), encode_sec=round(t_enc, 3))
            (meta_dir / f"{vid}.json").write_text(json.dumps(m), encoding="utf-8")

            ok += 1
            total_secs += meta.duration_sec
            total_frames += meta.sampled_frames
            if i % 25 == 0 or i <= 5:
                el = time.time() - t_start
                print(f"  [{i}/{len(todo)}] {vid} {emb.shape} "
                      f"dec {t_dec:.1f}s enc {t_enc:.1f}s | "
                      f"{total_secs / el:.1f}x realtime, "
                      f"eta {el / i * (len(todo) - i) / 60:.1f} min")
        except Exception as e:  # never let one bad file kill the run
            fail += 1
            print(f"  FAIL {vid}: {type(e).__name__}: {e}")

    el = time.time() - t_start
    print(f"\nencoded {ok}, failed {fail} in {el / 60:.1f} min")
    print(f"  {total_frames:.0f} frames, {total_secs / 60:.1f} min of video, "
          f"{total_secs / max(el, 1e-9):.1f}x realtime")
    return {"encoded": ok, "skipped": len(items) - len(todo), "failed": fail,
            "wall_sec": el, "video_sec": total_secs}


def collect(root: str | Path, split: str, limit: int | None) -> list[tuple[str, str]]:
    from .io_dataset import load_test_videos, load_train

    if split == "test":
        df = load_test_videos(root)
    else:
        df = load_train(root)
    items = list(zip(df["video_id"].astype(str), df["path"].astype(str)))
    return items[:limit] if limit else items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Train and Test")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=4096)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resize", type=int, default=224, help="pre-resize frames; SigLIP squashes to 224x224 anyway")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    items = collect(a.root, a.split, a.limit)
    run(items, Path(a.cache), a.model, a.fps, a.batch_size,
        a.max_frames, a.device, a.resize, a.overwrite)


if __name__ == "__main__":
    main()
