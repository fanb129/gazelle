#!/usr/bin/env bash
set -euo pipefail

# Formal VAT transferability pipeline on a source-video-disjoint train holdout.
# One invocation owns exactly one physical GPU and runs:
#   dense base -> support K25 -> learned-router K50 -> matched dense.
# Official test is intentionally deferred until a full-train refit is frozen.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 PYTHON_BIN GAZEFOLLOW_PRETRAIN_CHECKPOINT" >&2
  exit 2
fi

TRAIN_PYTHON_BIN=$1
GAZEFOLLOW_PRETRAIN_CHECKPOINT=$2
REPO_ROOT=$(git rev-parse --show-toplevel)
VAT_DATA_PATH=/newhome/fb/dataset/videoattentiontarget
TRAIN_SEED=3106
EXPECTED_GF_PRETRAIN_SHA256=26010ecc293cd51d2ce69f95e8111ca46632e425e1463b22fb0980aacf67fbef
VAT_TRAIN_ANNOTATION_SHA256=9831fe491277083dfc82cb843729a37e622c88c510e9c2c9e345158c03389e43
VAT_ASSIGNMENT_FINGERPRINT=42772300b78bea0cbeb1d288ab0bc9fb209651f2bf898ff9f8ee7abdfefcf793

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one physical GPU index." >&2
  exit 2
fi
if [[ ! -x "${TRAIN_PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${TRAIN_PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${GAZEFOLLOW_PRETRAIN_CHECKPOINT}" ]]; then
  echo "ERROR: GazeFollow pretrain checkpoint not found: ${GAZEFOLLOW_PRETRAIN_CHECKPOINT}" >&2
  exit 2
fi
ACTUAL_GF_PRETRAIN_SHA256=$("${TRAIN_PYTHON_BIN}" -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "${GAZEFOLLOW_PRETRAIN_CHECKPOINT}")
if [[ "${ACTUAL_GF_PRETRAIN_SHA256}" != "${EXPECTED_GF_PRETRAIN_SHA256}" ]]; then
  echo "ERROR: unexpected GazeFollow pretrain checkpoint SHA256." >&2
  echo "expected=${EXPECTED_GF_PRETRAIN_SHA256}" >&2
  echo "actual=${ACTUAL_GF_PRETRAIN_SHA256}" >&2
  exit 2
fi
if [[ ! -f "${VAT_DATA_PATH}/train_preprocessed.json" ]]; then
  echo "ERROR: VAT training annotations not found under ${VAT_DATA_PATH}." >&2
  exit 2
fi
ACTUAL_VAT_TRAIN_SHA256=$(sha256sum \
  "${VAT_DATA_PATH}/train_preprocessed.json" | awk '{print $1}')
if [[ "${ACTUAL_VAT_TRAIN_SHA256}" != "${VAT_TRAIN_ANNOTATION_SHA256}" ]]; then
  echo "ERROR: unexpected VAT train annotation SHA256." >&2
  echo "expected=${VAT_TRAIN_ANNOTATION_SHA256}" >&2
  echo "actual=${ACTUAL_VAT_TRAIN_SHA256}" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "ERROR: formal training requires a clean worktree." >&2
  git -C "${REPO_ROOT}" status --short >&2
  exit 2
fi

DENSE_BASE_DIR=${REPO_ROOT}/experiments/coverage_router/vat_splitclean_dense_seed3106
SUPPORT_DIR=${REPO_ROOT}/experiments/coverage_router/vat_splitclean_support_k025_seed3106
K50_DIR=${REPO_ROOT}/experiments/coverage_router/vat_splitclean_sparse_fixedrouter_decoder_k050_seed3106
DENSE_MATCHED_DIR=${REPO_ROOT}/experiments/coverage_router/vat_splitclean_dense_decoder_continue_seed3106

for RUN_DIR in \
  "${DENSE_BASE_DIR}" \
  "${SUPPORT_DIR}" \
  "${K50_DIR}" \
  "${DENSE_MATCHED_DIR}"; do
  if [[ -e "${RUN_DIR}" ]]; then
    echo "ERROR: refusing to reuse existing run directory: ${RUN_DIR}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
echo "VAT_PIPELINE seed=${TRAIN_SEED} gpu=${CUDA_VISIBLE_DEVICES}"
echo "runtime_git_commit=$(git rev-parse HEAD)"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu --format=csv

echo "STAGE vat_dense_base"
"${TRAIN_PYTHON_BIN}" -u scripts/train_coverage_router.py \
  --dataset vat \
  --model gazelle_dinov3_vitb16_inout \
  --data_path "${VAT_DATA_PATH}" \
  --vat_val_fraction 0.10 \
  --vat_split_seed 3106 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --init_ckpt "${GAZEFOLLOW_PRETRAIN_CHECKPOINT}" \
  --allow_init_without_split_provenance \
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
  --inout_loss_lambda 1.0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-5 \
  --lr_backbone 0 \
  --lr_inout 1e-3 \
  --weight_decay 0 \
  --max_epochs 8 \
  --batch_size 60 \
  --eval_batch_size 60 \
  --grad_accum_steps 1 \
  --n_workers 8 \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name vat_splitclean_dense_seed3106 \
  --run_dir "${DENSE_BASE_DIR}" \
  --seed 3106

"${TRAIN_PYTHON_BIN}" scripts/validate_coverage_router_run.py \
  --run_dir "${DENSE_BASE_DIR}" \
  --dataset vat \
  --seed 3106 \
  --router_stage support_pilot \
  --keep_ratio 1.0 \
  --epochs 8 \
  --batch_size 60 \
  --eval_batch_size 60 \
  --router_trainable 0 \
  --decoder_trainable 3416576 \
  --inout_trainable 33281 \
  --evaluation_split vat_train_source_video_holdout \
  --split_seed 3106 \
  --assignment_fingerprint "${VAT_ASSIGNMENT_FINGERPRINT}" \
  --source_annotation_sha256 "${VAT_TRAIN_ANNOTATION_SHA256}" \
  --train_sample_count 18607 \
  --eval_sample_count 4034 \
  --train_group_count 36 \
  --validation_group_count 4 \
  --init_checkpoint "${GAZEFOLLOW_PRETRAIN_CHECKPOINT}" \
  --selection_metric l2 \
  --selection_mode min \
  --lr_router 0 \
  --lr_decoder 1e-5 \
  --lr_inout 1e-3 \
  --router_warmup_epochs 0 \
  --grad_accum_steps 1 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --reinitialize_router_on_init false

echo "STAGE vat_support_k025"
"${TRAIN_PYTHON_BIN}" -u scripts/train_coverage_router.py \
  --dataset vat \
  --model gazelle_dinov3_vitb16_inout \
  --data_path "${VAT_DATA_PATH}" \
  --vat_val_fraction 0.10 \
  --vat_split_seed 3106 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --init_ckpt "${DENSE_BASE_DIR}/best_val_selection.pt" \
  --router_stage support_pilot \
  --route_after_block 5 \
  --keep_ratio 0.25 \
  --router_hidden_dim 256 \
  --router_temperature 1.0 \
  --escape_tokens 8 \
  --spatial_prior none \
  --fusion raw_concat \
  --router_warmup_epochs 0 \
  --heatmap_loss_weight 0 \
  --inout_loss_lambda 0 \
  --router_coverage_weight 1.0 \
  --router_budget_weight 0.05 \
  --router_entropy_weight 0 \
  --lr_router 1e-3 \
  --lr_decoder 0 \
  --lr_backbone 0 \
  --lr_inout 0 \
  --weight_decay 0 \
  --max_epochs 3 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 1 \
  --n_workers 8 \
  --log_iter 10 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name vat_splitclean_support_k025_seed3106 \
  --run_dir "${SUPPORT_DIR}" \
  --seed 3106

"${TRAIN_PYTHON_BIN}" scripts/validate_coverage_router_run.py \
  --run_dir "${SUPPORT_DIR}" \
  --dataset vat \
  --seed 3106 \
  --router_stage support_pilot \
  --keep_ratio 0.25 \
  --epochs 3 \
  --router_trainable 463105 \
  --decoder_trainable 0 \
  --inout_trainable 0 \
  --evaluation_split vat_train_source_video_holdout \
  --split_seed 3106 \
  --assignment_fingerprint "${VAT_ASSIGNMENT_FINGERPRINT}" \
  --source_annotation_sha256 "${VAT_TRAIN_ANNOTATION_SHA256}" \
  --train_sample_count 18607 \
  --eval_sample_count 4034 \
  --train_group_count 36 \
  --validation_group_count 4 \
  --init_checkpoint "${DENSE_BASE_DIR}/best_val_selection.pt" \
  --selection_metric routing_hard_coverage \
  --selection_mode max \
  --lr_router 1e-3 \
  --lr_decoder 0 \
  --lr_inout 0 \
  --router_warmup_epochs 0 \
  --grad_accum_steps 1 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --reinitialize_router_on_init false

echo "STAGE vat_learned_k50"
"${TRAIN_PYTHON_BIN}" -u scripts/train_coverage_router.py \
  --dataset vat \
  --model gazelle_dinov3_vitb16_inout \
  --data_path "${VAT_DATA_PATH}" \
  --vat_val_fraction 0.10 \
  --vat_split_seed 3106 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
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
  --inout_loss_lambda 1.0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_backbone 0 \
  --lr_inout 1e-3 \
  --weight_decay 0 \
  --max_epochs 5 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 2 \
  --n_workers 8 \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name vat_splitclean_sparse_fixedrouter_decoder_k050_seed3106 \
  --run_dir "${K50_DIR}" \
  --seed 3106

"${TRAIN_PYTHON_BIN}" scripts/validate_coverage_router_run.py \
  --run_dir "${K50_DIR}" \
  --dataset vat \
  --seed 3106 \
  --router_stage backbone_sparse \
  --keep_ratio 0.50 \
  --epochs 5 \
  --router_trainable 0 \
  --decoder_trainable 3416576 \
  --inout_trainable 33281 \
  --evaluation_split vat_train_source_video_holdout \
  --split_seed 3106 \
  --assignment_fingerprint "${VAT_ASSIGNMENT_FINGERPRINT}" \
  --source_annotation_sha256 "${VAT_TRAIN_ANNOTATION_SHA256}" \
  --train_sample_count 18607 \
  --eval_sample_count 4034 \
  --train_group_count 36 \
  --validation_group_count 4 \
  --init_checkpoint "${SUPPORT_DIR}/best_val_selection.pt" \
  --selection_metric l2 \
  --selection_mode min \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_inout 1e-3 \
  --router_warmup_epochs 3 \
  --grad_accum_steps 2 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --reinitialize_router_on_init false

echo "STAGE vat_matched_dense"
"${TRAIN_PYTHON_BIN}" -u scripts/train_coverage_router.py \
  --dataset vat \
  --model gazelle_dinov3_vitb16_inout \
  --data_path "${VAT_DATA_PATH}" \
  --vat_val_fraction 0.10 \
  --vat_split_seed 3106 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --init_ckpt "${DENSE_BASE_DIR}/best_val_selection.pt" \
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
  --inout_loss_lambda 1.0 \
  --router_coverage_weight 0 \
  --router_budget_weight 0 \
  --router_entropy_weight 0 \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_backbone 0 \
  --lr_inout 1e-3 \
  --weight_decay 0 \
  --max_epochs 5 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --grad_accum_steps 2 \
  --n_workers 8 \
  --clip_grad_norm 1.0 \
  --wandb_project GazeRoute \
  --wandb_mode online \
  --exp_name vat_splitclean_dense_decoder_continue_seed3106 \
  --run_dir "${DENSE_MATCHED_DIR}" \
  --seed 3106

"${TRAIN_PYTHON_BIN}" scripts/validate_coverage_router_run.py \
  --run_dir "${DENSE_MATCHED_DIR}" \
  --dataset vat \
  --seed 3106 \
  --router_stage support_pilot \
  --keep_ratio 1.0 \
  --epochs 5 \
  --router_trainable 0 \
  --decoder_trainable 3416576 \
  --inout_trainable 33281 \
  --evaluation_split vat_train_source_video_holdout \
  --split_seed 3106 \
  --assignment_fingerprint "${VAT_ASSIGNMENT_FINGERPRINT}" \
  --source_annotation_sha256 "${VAT_TRAIN_ANNOTATION_SHA256}" \
  --train_sample_count 18607 \
  --eval_sample_count 4034 \
  --train_group_count 36 \
  --validation_group_count 4 \
  --init_checkpoint "${DENSE_BASE_DIR}/best_val_selection.pt" \
  --selection_metric l2 \
  --selection_mode min \
  --lr_router 0 \
  --lr_decoder 1e-4 \
  --lr_inout 1e-3 \
  --router_warmup_epochs 0 \
  --grad_accum_steps 2 \
  --frame_sample_every 6 \
  --eval_frame_sample_every 6 \
  --reinitialize_router_on_init false

echo "VAT_HOLDOUT_PIPELINE complete"
