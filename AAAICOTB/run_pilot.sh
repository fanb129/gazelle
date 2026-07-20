#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-/home/fb/anaconda3/envs/py310/bin/python}"
DATA_PATH="${DATA_PATH:-/newhome/fb/dataset/videoattentiontarget}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/AAAIResults/COTB/pilot}"
GPU="${GPU:-0}"

# Fast mechanism pilot: initialize the exact historical GazeSpot architecture
# from its existing GF checkpoint, then compare lambda=0 and lambda=0.1 under
# the same grouped-frame loader.  This is not the final clean backbone claim.
GF_INIT_CKPT="${GF_INIT_CKPT:-$REPO/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt}"
V0_VAT_CKPT="${V0_VAT_CKPT:-$REPO/experiments/train_vat_vitb_v0/2026-03-20_22-30-50/epoch_7.pt}"
SPOT_VAT_CKPT="${SPOT_VAT_CKPT:-$REPO/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt}"

for required in "$PYTHON_BIN" "$DATA_PATH/train_preprocessed.json" "$DATA_PATH/test_preprocessed.json" "$GF_INIT_CKPT" "$V0_VAT_CKPT" "$SPOT_VAT_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m AAAICOTB.audit_dataset \
  --annotation "$DATA_PATH/train_preprocessed.json" \
  --output "$OUTPUT_ROOT/train_annotation_audit.json"

"$PYTHON_BIN" -m AAAICOTB.audit_dataset \
  --annotation "$DATA_PATH/test_preprocessed.json" \
  --output "$OUTPUT_ROOT/test_annotation_audit.json"

# Establish whether the failure exists in two already-trained systems.
"$PYTHON_BIN" -m AAAICOTB.evaluate \
  --data-path "$DATA_PATH" \
  --annotation "$DATA_PATH/test_preprocessed.json" \
  --checkpoint "$V0_VAT_CKPT" \
  --output-dir "$OUTPUT_ROOT/pretrained_v0" \
  --model-label pretrained_v0_448 \
  --model-source v0 \
  --batch-size-frames 4 \
  --bootstrap-iterations 2000

"$PYTHON_BIN" -m AAAICOTB.evaluate \
  --data-path "$DATA_PATH" \
  --annotation "$DATA_PATH/test_preprocessed.json" \
  --checkpoint "$SPOT_VAT_CKPT" \
  --output-dir "$OUTPUT_ROOT/pretrained_gazespot" \
  --model-label pretrained_gazespot_512 \
  --model-source current \
  --spatial-prior ggsf \
  --fusion sasa \
  --batch-size-frames 4 \
  --bootstrap-iterations 2000

COMMON_TRAIN=(
  --data-path "$DATA_PATH"
  --init-checkpoint "$GF_INIT_CKPT"
  --model gazelle_dinov3_vitb16_inout
  --spatial-prior ggsf
  --fusion sasa
  --epochs 2
  --batch-size-frames 6
  --frame-sample-every 6
  --validation-frame-sample-every 6
  --seed 3106
)

"$PYTHON_BIN" -m AAAICOTB.train \
  "${COMMON_TRAIN[@]}" \
  --bind-weight 0.0 \
  --output-dir "$OUTPUT_ROOT/grouped_control"

"$PYTHON_BIN" -m AAAICOTB.train \
  "${COMMON_TRAIN[@]}" \
  --bind-weight 0.10 \
  --output-dir "$OUTPUT_ROOT/cotb_w010"

COMMON_EVAL=(
  --data-path "$DATA_PATH"
  --annotation "$DATA_PATH/test_preprocessed.json"
  --model-source current
  --model gazelle_dinov3_vitb16_inout
  --spatial-prior ggsf
  --fusion sasa
  --batch-size-frames 4
  --bootstrap-iterations 5000
)

"$PYTHON_BIN" -m AAAICOTB.evaluate \
  "${COMMON_EVAL[@]}" \
  --checkpoint "$OUTPUT_ROOT/grouped_control/best_joint.pt" \
  --output-dir "$OUTPUT_ROOT/grouped_control_eval" \
  --model-label grouped_control

"$PYTHON_BIN" -m AAAICOTB.evaluate \
  "${COMMON_EVAL[@]}" \
  --checkpoint "$OUTPUT_ROOT/cotb_w010/best_joint.pt" \
  --output-dir "$OUTPUT_ROOT/cotb_w010_eval" \
  --model-label cotb_w010

"$PYTHON_BIN" -m AAAICOTB.compare \
  --control-pairs "$OUTPUT_ROOT/grouped_control_eval/pairs.csv" \
  --candidate-pairs "$OUTPUT_ROOT/cotb_w010_eval/pairs.csv" \
  --control-summary "$OUTPUT_ROOT/grouped_control_eval/summary.json" \
  --candidate-summary "$OUTPUT_ROOT/cotb_w010_eval/summary.json" \
  --output "$OUTPUT_ROOT/pilot_comparison.json" \
  --primary-subset far \
  --bootstrap-iterations 5000

echo "COTB pilot complete: $OUTPUT_ROOT/pilot_comparison.json"
