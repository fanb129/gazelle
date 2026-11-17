#!/usr/bin/env bash
# Run the two fixed-budget GPU jobs and the CPU comparison sequentially.
set -euo pipefail

python_path=$1
repository_path=$2
feature_cache_path=$3
experiment_root=$4
results_path=$5

cd "$repository_path"
mkdir -p "$experiment_root"

training_args=(
  --feature-cache "$feature_cache_path"
  --feature-report "$results_path/r2_feature_cache.json"
  --r1-report "$results_path/r1_split_and_diagnosis.json"
  --r1-review "$results_path/r1_manual_review.json"
  --plan "$repository_path/cscwd/r4_compact_plan.md"
  --epochs 5 --learning-rate 0.0001 --lambda-state 0.1
  --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0
  --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1
  --device cuda:0 --provenance "$results_path/RESULT_PROVENANCE.md"
)

printf 'Starting ordinary GRU; log: %s/r4_ordinary_gru.log\n' "$results_path"
"$python_path" -u cscwd/code/scripts/train_compact_joint.py \
  --model-id ordinary_gru "${training_args[@]}" \
  --output-dir "$experiment_root/ordinary_gru" \
  --report "$results_path/r4_ordinary_gru.json" \
  --summary "$results_path/r4_ordinary_gru.md" \
  > "$results_path/r4_ordinary_gru.log" 2>&1

printf 'Starting history validity model; log: %s/r4_history_validity.log\n' "$results_path"
"$python_path" -u cscwd/code/scripts/train_compact_joint.py \
  --model-id history_validity "${training_args[@]}" \
  --output-dir "$experiment_root/history_validity" \
  --report "$results_path/r4_history_validity.json" \
  --summary "$results_path/r4_history_validity.md" \
  > "$results_path/r4_history_validity.log" 2>&1

printf 'Comparing completed runs; log: %s/r4_validation_comparison.log\n' "$results_path"
"$python_path" -u cscwd/code/scripts/compare_compact_joint.py \
  --run-report "$results_path/r4_ordinary_gru.json" "$results_path/r4_history_validity.json" \
  --r1-report "$results_path/r1_split_and_diagnosis.json" \
  --plan "$repository_path/cscwd/r4_compact_plan.md" \
  --report "$results_path/r4_validation_comparison.json" \
  --summary "$results_path/r4_validation_comparison.md" \
  --selection-manifest "$results_path/r4_selection_manifest.json" \
  --provenance "$results_path/RESULT_PROVENANCE.md" \
  > "$results_path/r4_validation_comparison.log" 2>&1

printf 'R4 compact first round completed.\n'
