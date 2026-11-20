#!/usr/bin/env bash
# D4: one fresh current-frame control, per-epoch test, then reference comparison.
set -euo pipefail

python_path=$1
repository_path=$2
train_feature_cache=$3
test_feature_cache=$4
experiment_root=$5
results_path=$6

cd "$repository_path"
mkdir -p "$experiment_root"

printf 'Starting D4 current-frame training; log: %s/r4_current_frame_train.log\n' "$results_path"
"$python_path" -u cscwd/code/scripts/train_current_frame.py \
  --train-feature-cache "$train_feature_cache" \
  --train-feature-report "$results_path/r2_feature_cache.json" \
  --test-feature-cache "$test_feature_cache" \
  --test-feature-report "$results_path/r4_official_test_features.json" \
  --train-prediction-report "$results_path/r1_train_single_frame_cache.json" \
  --test-prediction-report "$results_path/t2_single_frame_cache.json" \
  --plan "$repository_path/cscwd/r4_current_frame_plan.md" \
  --epochs 5 --learning-rate 0.0001 --lambda-state 0.1 \
  --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0 \
  --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1 \
  --device cuda:0 \
  --output-dir "$experiment_root/current_frame" \
  --report "$results_path/r4_current_frame.json" \
  --summary "$results_path/r4_current_frame.md" \
  --provenance "$results_path/RESULT_PROVENANCE.md" \
  > "$results_path/r4_current_frame_train.log" 2>&1

printf 'Comparing D4 with original official results; log: %s/r4_current_frame_comparison.log\n' "$results_path"
"$python_path" -u cscwd/code/scripts/compare_current_frame.py \
  --run-report "$results_path/r4_current_frame.json" \
  --plan "$repository_path/cscwd/r4_current_frame_plan.md" \
  --reference-comparison "$results_path/r4_official_comparison.json" \
  --report "$results_path/r4_current_frame_comparison.json" \
  --summary "$results_path/r4_current_frame_comparison.md" \
  --selection-manifest "$results_path/r4_current_frame_selection.json" \
  --provenance "$results_path/RESULT_PROVENANCE.md" \
  > "$results_path/r4_current_frame_comparison.log" 2>&1

printf 'R4 current-frame round completed.\n'
