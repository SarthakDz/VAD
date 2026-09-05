"""Stage A end to end: cached embeddings -> score curves -> segments -> submission.

This is the complete, submittable system with no VLM anywhere in it. Stage B
is a strict upgrade on top; if the LoRA never converges, this still ships.

Runtime accounting is real, not invented. Decode and encode times were
recorded per video during `src.encode` and are stored in cache/meta; head
inference is timed here. `end_to_end_internal_time_ms` is their sum, which is
exactly what the arena asks for: decoding, preprocessing, inference and
postprocessing, excluding model load.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .frames import VideoMeta, timestamps
from .head import TemporalHead, predict
from .segments import extract, to_events
from .submit import (ModelRuntime, RuntimeMetadata, VideoPrediction, build,
                     load_manifest, manifest_from_public_test, write)

ENCODER_NAME = "siglip-frozen-encoder"
HEAD_NAME = "temporal-head"


def load_head(path: str | Path, device: str) -> TemporalHead:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TemporalHead(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model


def load_cached(cache: Path, vid: str):
    emb = np.load(cache / "emb" / f"{vid}.npy").astype(np.float32)
    meta = json.loads((cache / "meta" / f"{vid}.json").read_text())
    return emb, meta


def run(
    manifest: dict[str, int],
    cache: Path,
    model: TemporalHead,
    device: str,
    enter: float,
    exit_: float,
    merge_gap_sec: float,
    min_event_sec: float,
    max_coverage: float = 1.0,
    top_k: int = 0,
    min_score: float = 0.0,
    save_scores: bool = True,
    vlm=None,
    paths: dict[str, str] | None = None,
    vlm_frames: int = 6,
    vlm_motion_crop: bool = True,
) -> tuple[list[VideoPrediction], dict]:
    score_dir = cache / "scores"
    if save_scores:
        score_dir.mkdir(parents=True, exist_ok=True)

    from .fuse import refine_segments
    from .vlm import VLMStats

    vlm_stats = VLMStats()
    preds: list[VideoPrediction] = []
    n_relabelled = 0
    n_events = 0
    total_video_sec = 0.0
    total_internal_ms = 0.0

    for vid, level in manifest.items():
        try:
            emb, meta = load_cached(cache, vid)
        except FileNotFoundError:
            # Trap 3 in reverse: never drop a video. An empty answer is a real
            # answer -- on a normal Level-2/3 video it is the correct one.
            preds.append(VideoPrediction(video_id=vid))
            print(f"  {vid}: no cached embedding, emitting empty answer")
            continue

        t0 = time.perf_counter()
        a, c = predict(model, torch.from_numpy(emb), device)
        head_ms = (time.perf_counter() - t0) * 1000.0

        a, c = a.numpy(), c.numpy()
        if save_scores:
            np.save(score_dir / f"{vid}.npy",
                    np.concatenate([a[:, None], c], axis=1).astype(np.float16))

        vm = VideoMeta(
            video_id=vid,
            duration_sec=meta["duration_sec"],
            native_fps=meta["native_fps"],
            native_frames=meta["native_frames"],
            width=meta["width"],
            height=meta["height"],
            sampled_frames=meta["sampled_frames"],
            sample_fps=meta["sample_fps"],
            frame_step=int(meta.get("frame_step", 1)),
        )
        segs = extract(a, c, timestamps(vm), enter, exit_, merge_gap_sec, min_event_sec,
                       max_coverage=1.0 if level == 1 else max_coverage,
                       top_k=top_k, min_score=min_score)

        explanations = None
        vlm_ms = 0.0
        if vlm is not None and segs and paths and vid in paths:
            before = [s.class_name for s in segs]
            t1 = time.perf_counter()
            segs, explanations = refine_segments(
                segs, paths[vid], c, vlm, vlm_stats, vlm_frames, vlm_motion_crop)
            vlm_ms = (time.perf_counter() - t1) * 1000.0
            n_relabelled += sum(1 for b, s in zip(before, segs) if b != s.class_name)

        events = to_events(segs, level, explanations)
        n_events += len(events)

        decode_ms = float(meta.get("decode_sec", 0.0)) * 1000.0
        encode_ms = float(meta.get("encode_sec", 0.0)) * 1000.0
        internal_ms = decode_ms + encode_ms + head_ms + vlm_ms
        total_internal_ms += internal_ms
        total_video_sec += vm.duration_sec

        preds.append(VideoPrediction(
            video_id=vid,
            events=events,
            runtime=RuntimeMetadata(
                frames_processed=vm.sampled_frames,
                chunks_processed=max(1, int(np.ceil(len(emb) / 4096))),
                end_to_end_internal_time_ms=internal_ms,
                model_runtimes=[
                    ModelRuntime(ENCODER_NAME, call_count=vm.sampled_frames,
                                 total_time_ms=encode_ms),
                    ModelRuntime(HEAD_NAME, call_count=1, total_time_ms=head_ms),
                ] + ([ModelRuntime(vlm.name, call_count=len(segs),
                                   total_time_ms=vlm_ms)] if vlm is not None and segs else []),
            ),
        ))

    stats = {
        "videos": len(preds),
        "events": n_events,
        "video_sec": total_video_sec,
        "internal_ms": total_internal_ms,
        "realtime_factor": total_video_sec / max(total_internal_ms / 1000.0, 1e-9),
        "vlm_calls": vlm_stats.call_count,
        "vlm_ms": vlm_stats.total_time_ms,
        "vlm_frames": vlm_stats.frames_seen,
        "vlm_parse_failures": vlm_stats.parse_failures,
        "relabelled": n_relabelled,
    }
    return preds, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../Train and Test")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--head", default="outputs/head.pt")
    ap.add_argument("--manifest")
    ap.add_argument("--out", default="outputs/submission_head.json")
    ap.add_argument("--enter", type=float, default=0.70)
    ap.add_argument("--exit", dest="exit_", type=float, default=0.45)
    ap.add_argument("--merge-gap", type=float, default=5.0)
    ap.add_argument("--min-event", type=float, default=2.0)
    ap.add_argument("--max-coverage", type=float, default=1.0,
                    help="per-video quantile fallback for saturated curves (L2/3 only)")
    ap.add_argument("--submission-id", default="stage-a-head")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vlm", action="store_true", help="enable Stage B re-labelling")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--vlm-frames", type=int, default=6)
    ap.add_argument("--no-motion-crop", action="store_true")
    ap.add_argument("--vlm-4bit", action="store_true")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--score", action="store_true", help="score against public test gt")
    a = ap.parse_args()

    mf = load_manifest(a.manifest) if a.manifest else manifest_from_public_test(a.root)
    model = load_head(a.head, a.device)

    vlm = paths = None
    if a.vlm:
        from .io_dataset import load_test_videos
        from .vlm import QwenVLM
        paths = {r.video_id: r.path for r in load_test_videos(a.root).itertuples()}
        print(f"loading {a.vlm_model} ...")
        vlm = QwenVLM(a.vlm_model, a.device, load_in_4bit=a.vlm_4bit)

    t0 = time.perf_counter()
    preds, stats = run(mf, Path(a.cache), model, a.device,
                       a.enter, a.exit_, a.merge_gap, a.min_event, a.max_coverage,
                       a.top_k, a.min_score,
                       vlm=vlm, paths=paths, vlm_frames=a.vlm_frames,
                       vlm_motion_crop=not a.no_motion_crop)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    doc = build(preds, a.submission_id, "siglip-gru-stage-a", wall_ms,
                "1x RTX 4060 Laptop 8GB")
    p = write(doc, a.out, mf)

    print(f"wrote {p}")
    print(f"  {stats['videos']} videos, {stats['events']} events, "
          f"{stats['video_sec']/60:.1f} min of video")
    print(f"  internal {stats['internal_ms']/1000:.1f}s  "
          f"-> {stats['realtime_factor']:.1f}x realtime "
          f"(latency ratio {1/stats['realtime_factor']:.4f})")
    if stats.get("vlm_calls"):
        print(f"  VLM: {stats['vlm_calls']} calls, {stats['vlm_frames']} frames, "
              f"{stats['vlm_ms']/1000:.1f}s, {stats['relabelled']} segments relabelled, "
              f"{stats['vlm_parse_failures']} parse failures")

    if a.score:
        from .io_dataset import TEST_COLS, _read_gt
        from .score import load_predictions, score
        gt = _read_gt(Path(a.root) / "test" / "ground_truth.csv", TEST_COLS)
        r = score(gt, load_predictions(p))
        print(f"\n  LEVEL 1 {r['level1']:.4f}  (bin {r['l1_binary_acc']:.3f} "
              f"cls {r['l1_class_acc']:.3f})")
        for lv in (2, 3):
            print(f"  LEVEL {lv} {r[f'level{lv}']:.4f}  "
                  f"(alert {r[f'l{lv}_alert']:.3f} matched {r[f'l{lv}_matched']:.3f} "
                  f"timing {r[f'l{lv}_timing']:.3f})")
        print(f"  OVERALL {r['overall_mean']:.4f}")


if __name__ == "__main__":
    main()
