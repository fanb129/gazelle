#!/usr/bin/env bash
# Fixed D3 state supervision: two fresh models, per-epoch test, then comparison.
set -euo pipefail

python_path=$1
repository_path=$2
train_feature_cache=$3
test_feature_cache=$4
experiment_root=$5
results_path=$6

cd "$repository_path"
mkdir -p "$experiment_root"

training_args=(
  --train-feature-cache "$train_feature_cache"
  --train-feature-report "$results_path/r2_feature_cache.json"
  --test-feature-cache "$test_feature_cache"
  --test-feature-report "$results_path/r4_official_test_features.json"
  --train-prediction-report "$results_path/r1_train_single_frame_cache.json"
  --test-prediction-report "$results_path/t2_single_frame_cache.json"
  --plan "$repository_path/cscwd/r4_state_supervision_plan.md"
  --epochs 5 --learning-rate 0.0001 --lambda-state 0.1
  --lambda-time 0.1 --response-frames 3
  --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0
  --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1
  --device cuda:0 --provenance "$results_path/RESULT_PROVENANCE.md"
)

for model_id in ordinary_gru history_validity; do
  printf 'Starting D3 %s; log: %s/r4_state_supervised_%s.log\n' "$model_id" "$results_path" "$model_id"
  "$python_path" -u cscwd/code/scripts/train_state_supervised.py \
    --model-id "$model_id" "${training_args[@]}" \
    --output-dir "$experiment_root/$model_id" \
    --report "$results_path/r4_state_supervised_$model_id.json" \
    --summary "$results_path/r4_state_supervised_$model_id.md" \
    > "$results_path/r4_state_supervised_$model_id.log" 2>&1
done

printf 'Comparing D3 with original outputs; log: %s/r4_state_supervised_comparison.log\n' "$results_path"
"$python_path" -u cscwd/code/scripts/compare_state_supervised.py \
  --run-report "$results_path/r4_state_supervised_ordinary_gru.json" "$results_path/r4_state_supervised_history_validity.json" \
  --plan "$repository_path/cscwd/r4_state_supervision_plan.md" \
  --previous-comparison "$results_path/r4_official_comparison.json" \
  --report "$results_path/r4_state_supervised_comparison.json" \
  --summary "$results_path/r4_state_supervised_comparison.md" \
  --selection-manifest "$results_path/r4_state_supervised_selection.json" \
  --provenance "$results_path/RESULT_PROVENANCE.md" \
  > "$results_path/r4_state_supervised_comparison.log" 2>&1

printf 'R4 state-supervised round completed.\n'
