#!/usr/bin/env bash
set -euo pipefail

REPO=/home/fb/src/paper/gazelleV1
PYTHON=/home/fb/anaconda3/envs/py310/bin/python
DATA=/newhome/fb/dataset/videoattentiontarget
INIT=/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt
RUN_DIR=/home/fb/src/paper/gazelleV1/AAAIResults/P1a/prior_residual_vat_seed3106
RESULT_DIR=/home/fb/src/paper/gazelleV1/AAAIResults/P1a
PRIOR_DIR=/home/fb/src/paper/gazelleV1/AAAIResults/P1a/train_prior_1000

if [[ "$(pwd)" != "$REPO" ]]; then
  echo "Run this script from $REPO" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$RESULT_DIR" "$PRIOR_DIR"

"$PYTHON" -u AAAIScripts/audit_sasa_ggsf.py \
  --dataset vat \
  --data-path "$DATA" \
  --json-path "$DATA/train_preprocessed.json" \
  --checkpoint "$INIT" \
  --model-source current \
  --model gazelle_dinov3_vitb16_inout \
  --max-samples 1000 \
  --sampling uniform \
  --sampling-seed 3106 \
  --device cuda:0 \
  --output-dir "$PRIOR_DIR"

"$PYTHON" -u AAAIScripts/train_person_router.py \
  --dataset vat \
  --data-path "$DATA" \
  --init-checkpoint "$INIT" \
  --allow-legacy-sasa-ggsf \
  --train-scope router_only \
  --validation-from-train \
  --validation-fraction 0.1 \
  --validation-seed 3106 \
  --router-prior-json "$PRIOR_DIR/results.json" \
  --router-residual-scale 1.0 \
  --epochs 4 \
  --batch-size 60 \
  --frame-sample-every 6 \
  --lr 1e-3 \
  --seed 3106 \
  --device cuda:0 \
  --output-dir "$RUN_DIR"

"$PYTHON" -u AAAIScripts/failure_taxonomy.py \
  --dataset vat \
  --data-path "$DATA" \
  --json-path "$DATA/test_preprocessed.json" \
  --checkpoint "$RUN_DIR/initial.pt" \
  --model-source aaai_router \
  --model aaai_person_router_dinov3_vitb16_inout \
  --model-label p1a_static_prior \
  --device cuda:0 \
  --output-prefix "$RESULT_DIR/vat_static_prior_full"

"$PYTHON" -u AAAIScripts/failure_taxonomy.py \
  --dataset vat \
  --data-path "$DATA" \
  --json-path "$DATA/test_preprocessed.json" \
  --checkpoint "$RUN_DIR/best.pt" \
  --model-source aaai_router \
  --model aaai_person_router_dinov3_vitb16_inout \
  --model-label p1a_prior_residual \
  --device cuda:0 \
  --output-prefix "$RESULT_DIR/vat_prior_residual_full"

"$PYTHON" -u AAAIScripts/compare_failure_reports.py \
  --reference-records "$RESULT_DIR/vat_static_prior_full.records.csv" \
  --candidate-records "$RESULT_DIR/vat_prior_residual_full.records.csv" \
  --bootstrap-iterations 2000 \
  --seed 3106 \
  --output-prefix "$RESULT_DIR/static_prior_vs_prior_residual"

"$PYTHON" -u AAAIScripts/audit_sasa_ggsf.py \
  --dataset vat \
  --data-path "$DATA" \
  --json-path "$DATA/test_preprocessed.json" \
  --checkpoint "$RUN_DIR/best.pt" \
  --model-source aaai_router \
  --model aaai_person_router_dinov3_vitb16_inout \
  --max-samples 1000 \
  --sampling uniform \
  --sampling-seed 3106 \
  --device cuda:0 \
  --output-dir "$RESULT_DIR/router_audit"

echo "P1a VAT pilot and evaluation complete: $RESULT_DIR"
