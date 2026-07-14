#!/usr/bin/env bash
set -euo pipefail

if [[ "$(pwd)" != "/home/fb/src/paper/gazelleV1" ]]; then
  echo "Run this script from /home/fb/src/paper/gazelleV1" >&2
  exit 2
fi

mkdir -p /home/fb/src/paper/gazelleV1/AAAIResults/P2/transition_reliability

/home/fb/anaconda3/envs/py310/bin/python -u AAAIScripts/p2_transition_reliability_audit.py \
  --annotation-json /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json \
  --records baseline_v0_448=/home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_base_full.records.csv \
  --records learned_sasa_ggsf_512=/home/fb/src/paper/gazelleV1/AAAIResults/P0/vat_spot_full.records.csv \
  --records p1a_static_prior_512=/home/fb/src/paper/gazelleV1/AAAIResults/P1a/vat_static_prior_full.records.csv \
  --records p1a_prior_residual_512=/home/fb/src/paper/gazelleV1/AAAIResults/P1a/vat_prior_residual_full.records.csv \
  --iou-threshold 0.3 \
  --max-frame-gap 2 \
  --ece-bins 15 \
  --bootstrap-iterations 2000 \
  --seed 3106 \
  --output-dir /home/fb/src/paper/gazelleV1/AAAIResults/P2/transition_reliability

echo "P2 transition reliability audit complete."
