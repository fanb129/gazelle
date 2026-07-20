#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-/home/fb/anaconda3/envs/py310/bin/python}"
DATA_PATH="${DATA_PATH:-/newhome/fb/dataset/videoattentiontarget}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/AAAIResults/COTB/full}"
GPU="${GPU:-0}"
BIND_WEIGHT="${BIND_WEIGHT:-0.10}"
GF_RAW_CKPT="${GF_RAW_CKPT:-$REPO/AAAIResults/COTB/gf_rawconcat/best.pt}"
SEEDS=(3106 3407 4508)

for required in "$PYTHON_BIN" "$DATA_PATH/train_preprocessed.json" "$DATA_PATH/test_preprocessed.json" "$GF_RAW_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    echo "Train the clean GazeFollow raw-concat initialization first; see SERVER_COMMANDS.md." >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

for seed in "${SEEDS[@]}"; do
  CONTROL_DIR="$OUTPUT_ROOT/seed${seed}_control"
  COTB_DIR="$OUTPUT_ROOT/seed${seed}_cotb_w${BIND_WEIGHT/./}"

  COMMON_TRAIN=(
    --data-path "$DATA_PATH"
    --init-checkpoint "$GF_RAW_CKPT"
    --model gazelle_dinov3_vitb16_inout
    --spatial-prior none
    --fusion raw_concat
    --epochs 8
    --batch-size-frames 6
    --frame-sample-every 6
    --validation-frame-sample-every 6
    --validation-seed 9102
    --seed "$seed"
  )

  "$PYTHON_BIN" -m AAAICOTB.train \
    "${COMMON_TRAIN[@]}" \
    --bind-weight 0.0 \
    --output-dir "$CONTROL_DIR"

  "$PYTHON_BIN" -m AAAICOTB.train \
    "${COMMON_TRAIN[@]}" \
    --bind-weight "$BIND_WEIGHT" \
    --output-dir "$COTB_DIR"

  COMMON_EVAL=(
    --data-path "$DATA_PATH"
    --annotation "$DATA_PATH/test_preprocessed.json"
    --model-source current
    --model gazelle_dinov3_vitb16_inout
    --spatial-prior none
    --fusion raw_concat
    --batch-size-frames 4
    --bootstrap-iterations 5000
  )

  "$PYTHON_BIN" -m AAAICOTB.evaluate \
    "${COMMON_EVAL[@]}" \
    --checkpoint "$CONTROL_DIR/best_joint.pt" \
    --output-dir "${CONTROL_DIR}_eval" \
    --model-label "control_seed${seed}"

  "$PYTHON_BIN" -m AAAICOTB.evaluate \
    "${COMMON_EVAL[@]}" \
    --checkpoint "$COTB_DIR/best_joint.pt" \
    --output-dir "${COTB_DIR}_eval" \
    --model-label "cotb_seed${seed}"

  "$PYTHON_BIN" -m AAAICOTB.compare \
    --control-pairs "${CONTROL_DIR}_eval/pairs.csv" \
    --candidate-pairs "${COTB_DIR}_eval/pairs.csv" \
    --control-summary "${CONTROL_DIR}_eval/summary.json" \
    --candidate-summary "${COTB_DIR}_eval/summary.json" \
    --output "$OUTPUT_ROOT/seed${seed}_comparison.json" \
    --primary-subset far \
    --bootstrap-iterations 5000
done

"$PYTHON_BIN" -m AAAICOTB.summarize_seeds \
  --comparison "$OUTPUT_ROOT/seed3106_comparison.json" \
  --comparison "$OUTPUT_ROOT/seed3407_comparison.json" \
  --comparison "$OUTPUT_ROOT/seed4508_comparison.json" \
  --output "$OUTPUT_ROOT/three_seed_summary.json"

echo "COTB full confirmation complete: $OUTPUT_ROOT/three_seed_summary.json"
