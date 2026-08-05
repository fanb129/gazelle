#!/usr/bin/env bash
set -euo pipefail

# Fixed-base downstream-seed robustness pipeline for GazeFollow.
# One invocation owns exactly one physical GPU and runs:
#   support K25 -> learned-router K50 -> matched dense continuation.

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 TRAIN_SEED PYTHON_BIN BASE_DENSE_CHECKPOINT" >&2
  exit 2
fi

TRAIN_SEED=$1
TRAIN_PYTHON_BIN=$2
BASE_DENSE_CHECKPOINT=$3
REPO_ROOT=$(git rev-parse --show-toplevel)
GF_DATA_PATH=/newhome/fb/dataset/gazefollow_extended
GF_ASSIGNMENT_FINGERPRINT=817d9f2200ebc9881f6d2318cdb60d7d8d7af394f98f0e3461a6d8bd2c26b5e3
EXPECTED_BASE_DENSE_SHA256=4752899dbedde10cf211684c07ef3fee9acb2db990f031e484eed07b8f5d911d

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one physical GPU index." >&2
  exit 2
fi
if [[ ! "${TRAIN_SEED}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: TRAIN_SEED must be a positive integer." >&2
  exit 2
fi
if [[ ! -x "${TRAIN_PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${TRAIN_PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${BASE_DENSE_CHECKPOINT}" ]]; then
  echo "ERROR: dense checkpoint not found: ${BASE_DENSE_CHECKPOINT}" >&2
  exit 2
fi
ACTUAL_BASE_DENSE_SHA256=$(sha256sum "${BASE_DENSE_CHECKPOINT}" | awk '{print $1}')
if [[ "${ACTUAL_BASE_DENSE_SHA256}" != "${EXPECTED_BASE_DENSE_SHA256}" ]]; then
  echo "ERROR: unexpected GazeFollow dense base checkpoint SHA256." >&2
  echo "expected=${EXPECTED_BASE_DENSE_SHA256}" >&2
  echo "actual=${ACTUAL_BASE_DENSE_SHA256}" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ERROR: formal training requires a clean worktree." >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 2
fi

SUPPORT_DIR=${REPO_ROOT}/experiments/coverage_router/gf_splitclean_support_k025_seed${TRAIN_SEED}
K50_DIR=${REPO_ROOT}/experiments/coverage_router/gf_splitclean_sparse_fixedrouter_decoder_k050_seed${TRAIN_SEED}
DENSE_DIR=${REPO_ROOT}/experiments/coverage_router/gf_splitclean_dense_decoder_continue_seed${TRAIN_SEED}

for RUN_DIR in "${SUPPORT_DIR}" "${K50_DIR}" "${DENSE_DIR}"; do
  if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: refusing to reuse existing run directory: ${RUN_DIR}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
echo "MULTISEED_PIPELINE seed=${TRAIN_SEED} gpu=${CUDA_VISIBLE_DEVICES}"
echo "runtime_git_commit=$(git rev-parse HEAD)"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu --format=csv

echo "STAGE support_k025 seed=${TRAIN_SEED}"
"${TRAIN_PYTHON_BIN}" -u scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path "${GF_DATA_PATH}" \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --init_ckpt "${BASE_DENSE_CHECKPOINT}" \
  --reinitialize_router_on_init \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --router_hidden_dim 256 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --spatial_prior none \
  --fusion raw_concat \
  --heatmap_loss_weight 0 \
  --inout_loss_lambda 0 \
  --router_coverage_weight 1.0 \
  --router_budget_weight 0.05 \
  --router_entropy_weight 0 \
  --lr_router 1e-3 \
  --lr_decoder 0 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --router_warmup_epochs 0 \
  --weight_decay 0 \
  --max_epochs 3 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 1 \
  --n_workers 8 \
  --log_iter 10 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name "gf_splitclean_support_k025_seed${TRAIN_SEED}" \
  --run_dir "${SUPPORT_DIR}" \
  --seed "${TRAIN_SEED}"

"${TRAIN_PYTHON_BIN}" scripts/validate_coverage_router_run.py \
  --run_dir "${SUPPORT_DIR}" \
  --dataset gazefollow \
  --seed "${TRAIN_SEED}" \
  --router_stage support_pilot \
  --keep_ratio 0.25 \
  --epochs 3 \
  --router_trainable 463105 \
  --decoder_trainable 0 \
  --inout_trainable 0 \
  --evaluation_split gazefollow_train_holdout \
  --split_seed 3106 \
  --init_checkpoint "${BASE_DENSE_CHECKPOINT}" \
  --selection_metric routing_hard_coverage \
  --selection_mode max \
  --lr_router 1e-3 \
  --lr_decoder 0 \
  --lr_inout 0 \
  --router_warmup_epochs 0 \
  --grad_accum_steps 1 \
  --assignment_fingerprint "${GF_ASSIGNMENT_FINGERPRINT}" \
  --reinitialize_router_on_init true

echo "STAGE learned_k50 seed=${TRAIN_SEED}"
"${TRAIN_PYTHON_BIN}" -u scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path "${GF_DATA_PATH}" \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --init_ckpt "${SUPPORT_DIR}/best_val_selection.pt" \
  --router_stage backbone_sparse \
  --route_after_block 5 \
  --keep_ratio 0.50 \
  --router_hidden_dim 256 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --spatial_prior none \
  --fusion raw_concat \
  --router_warmup_epochs 3 \
  --heatmap_loss_weight 1.0 \
  --inout_loss_lambda 0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --weight_decay 0 \
  --max_epochs 5 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 2 \
  --n_workers 8 \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name "gf_splitclean_sparse_fixedrouter_decoder_k050_seed${TRAIN_SEED}" \
  --run_dir "${K50_DIR}" \
  --seed "${TRAIN_SEED}"

"${TRAIN_PYTHON_BIN}" scripts/validate_coverage_router_run.py \
  --run_dir "${K50_DIR}" \
  --dataset gazefollow \
  --seed "${TRAIN_SEED}" \
  --router_stage backbone_sparse \
  --keep_ratio 0.50 \
  --epochs 5 \
  --router_trainable 0 \
  --decoder_trainable 3416576 \
  --inout_trainable 0 \
  --evaluation_split gazefollow_train_holdout \
  --split_seed 3106 \
  --init_checkpoint "${SUPPORT_DIR}/best_val_selection.pt" \
  --selection_metric avg_l2 \
  --selection_mode min \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_inout 0 \
  --router_warmup_epochs 3 \
  --grad_accum_steps 2 \
  --assignment_fingerprint "${GF_ASSIGNMENT_FINGERPRINT}" \
  --reinitialize_router_on_init false

echo "STAGE matched_dense seed=${TRAIN_SEED}"
"${TRAIN_PYTHON_BIN}" -u scripts/train_coverage_router.py \
  --dataset gazefollow \
  --model gazelle_dinov3_vitb16 \
  --data_path "${GF_DATA_PATH}" \
  --gf_val_fraction 0.10 \
  --gf_split_seed 3106 \
  --init_ckpt "${BASE_DENSE_CHECKPOINT}" \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 1.0 \
  --router_hidden_dim 256 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --spatial_prior none \
  --fusion raw_concat \
  --router_warmup_epochs 0 \
  --heatmap_loss_weight 1.0 \
  --inout_loss_lambda 0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --weight_decay 0 \
  --max_epochs 5 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 2 \
  --n_workers 8 \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name "gf_splitclean_dense_decoder_continue_seed${TRAIN_SEED}" \
  --run_dir "${DENSE_DIR}" \
  --seed "${TRAIN_SEED}"

"${TRAIN_PYTHON_BIN}" scripts/validate_coverage_router_run.py \
  --run_dir "${DENSE_DIR}" \
  --dataset gazefollow \
  --seed "${TRAIN_SEED}" \
  --router_stage support_pilot \
  --keep_ratio 1.0 \
  --epochs 5 \
  --router_trainable 0 \
  --decoder_trainable 3416576 \
  --inout_trainable 0 \
  --evaluation_split gazefollow_train_holdout \
  --split_seed 3106 \
  --init_checkpoint "${BASE_DENSE_CHECKPOINT}" \
  --selection_metric avg_l2 \
  --selection_mode min \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_inout 0 \
  --router_warmup_epochs 0 \
  --grad_accum_steps 2 \
  --assignment_fingerprint "${GF_ASSIGNMENT_FINGERPRINT}" \
  --reinitialize_router_on_init false

echo "MULTISEED_PIPELINE complete seed=${TRAIN_SEED}"
