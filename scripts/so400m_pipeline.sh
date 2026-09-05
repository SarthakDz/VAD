#!/usr/bin/env bash
# Re-run the whole Stage A on so400m features, then score against the calibrated
# scorer. Encoder swap only -- no architecture change, so any gain is
# attributable to the representation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
C=cache_so400m
M=google/siglip-so400m-patch14-384

echo "== 1. encode train (test assumed done) =="
$PY -m src.encode --split train --cache $C --model $M --resize 384 --batch-size 16 \
  2>&1 | grep -E "videos,|encoded|FAIL" || true

echo "== 2. clip classifier =="
$PY -m src.train_clip --cache $C --out outputs/clip_so.pt 2>&1 | tail -18

echo "== 3. temporal head =="
$PY -m src.train_head --cache $C --out outputs/head_so.pt --epochs 15 2>&1 | grep -E "^ClipBank|^best"

echo "== 4. score both =="
$PY -m src.infer_head --cache $C --head outputs/head_so.pt --manifest data/manifest.json \
  --enter 0.92 --exit 0.30 --merge-gap 20 --min-event 3 --out outputs/sub_so_head.json >/dev/null
$PY -m src.calibrated --pred outputs/sub_so_head.json
$PY -m src.infer_clip --cache $C --clip outputs/clip_so.pt --out outputs/sub_so_clip.json --score \
  2>&1 | tail -3
echo "baseline to beat: 47.0"
