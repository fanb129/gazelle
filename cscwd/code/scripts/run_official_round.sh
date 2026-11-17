#!/usr/bin/env bash
# Full official train, complete official test after each epoch, then comparison.
set -euo pipefail

python_path=$1
repository_path=$2
train_feature_cache=$3
test_feature_cache=$4
experiment_root=$5
results_path=$6

cd "$repository_path"
mkdir -p "$experiment_root"

if [[ ! -f "$results_path/r4_official_test_features.json" ]]; then
  printf 'Caching frozen official test features; log: %s/r4_official_test_features.log\n' "$results_path"
  "$python_path" -u cscwd/code/scripts/cache_official_test_features.py \
    --data-path /newhome/fb/dataset/videoattentiontarget \
    --prediction-report "$results_path/t2_single_frame_cache.json" \
    --train-feature-report "$results_path/r2_feature_cache.json" \
    --checkpoint "$repository_path/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt" \
    --output-dir "$test_feature_cache" \
    --report "$results_path/r4_official_test_features.json" \
    --summary "$results_path/r4_official_test_features.md" \
    --provenance "$results_path/RESULT_PROVENANCE.md" \
    --batch-size 8 --num-workers 4 --device cuda:0 \
    > "$results_path/r4_official_test_features.log" 2>&1
else
  printf 'Using completed test feature report; training will verify its cache and sources.\n'
fi

training_args=(
  --train-feature-cache "$train_feature_cache"
  --train-feature-report "$results_path/r2_feature_cache.json"
  --test-feature-cache "$test_feature_cache"
  --test-feature-report "$results_path/r4_official_test_features.json"
  --train-prediction-report "$results_path/r1_train_single_frame_cache.json"
  --test-prediction-report "$results_path/t2_single_frame_cache.json"
  --plan "$repository_path/cscwd/r4_official_train_test_plan.md"
  --epochs 5 --learning-rate 0.0001 --lambda-state 0.1
  --hidden-dim 128 --clip-frames 16 --batch-size 8 --seed 0
  --weight-decay 0.01 --gradient-clip-norm 1.0 --temperature 0.1
  --device cuda:0 --provenance "$results_path/RESULT_PROVENANCE.md"
)

for model_id in ordinary_gru history_validity; do
  printf 'Starting %s; log: %s/r4_official_%s.log\n' "$model_id" "$results_path" "$model_id"
  "$python_path" -u cscwd/code/scripts/train_official_joint.py \
    --model-id "$model_id" "${training_args[@]}" \
    --output-dir "$experiment_root/$model_id" \
    --report "$results_path/r4_official_$model_id.json" \
    --summary "$results_path/r4_official_$model_id.md" \
    > "$results_path/r4_official_$model_id.log" 2>&1
done

printf 'Comparing completed official runs; log: %s/r4_official_comparison.log\n' "$results_path"
"$python_path" -u cscwd/code/scripts/compare_official_joint.py \
  --run-report "$results_path/r4_official_ordinary_gru.json" "$results_path/r4_official_history_validity.json" \
  --plan "$repository_path/cscwd/r4_official_train_test_plan.md" \
  --report "$results_path/r4_official_comparison.json" \
  --summary "$results_path/r4_official_comparison.md" \
  --selection-manifest "$results_path/r4_official_selection.json" \
  --provenance "$results_path/RESULT_PROVENANCE.md" \
  > "$results_path/r4_official_comparison.log" 2>&1

printf 'R4 official train/test round completed.\n'
