#!/usr/bin/env bash
set -euo pipefail

# Formal GazeFollow pipeline:
#   dense15 -> { support3 -> K50-5, matched-dense-5 } -> two one-shot tests
# The two branches and the two official evaluations use separate physical GPUs.

usage() {
  echo "Usage: $0 PYTHON_BIN DENSE_GPU ROUTED_GPU" >&2
  echo "Example: $0 /home/fb/anaconda3/envs/py310/bin/python 1 2" >&2
}

die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# -eq 3 ]] || { usage; exit 2; }
PYTHON_BIN=$1
DENSE_GPU=$2
ROUTED_GPU=$3
[[ -x "${PYTHON_BIN}" ]] || die "Python executable not found: ${PYTHON_BIN}"
[[ "${DENSE_GPU}" =~ ^[0-9]+$ ]] || die "DENSE_GPU must be one integer"
[[ "${ROUTED_GPU}" =~ ^[0-9]+$ ]] || die "ROUTED_GPU must be one integer"
[[ "${DENSE_GPU}" != "${ROUTED_GPU}" ]] || die "DENSE_GPU and ROUTED_GPU must differ"

ROOT=$(git rev-parse --show-toplevel)
COMMIT=$(git -C "${ROOT}" rev-parse HEAD)
DATA=/newhome/fb/dataset/gazefollow_extended
TRAIN_JSON=${DATA}/train_preprocessed.json
TEST_JSON=${DATA}/test_preprocessed.json
DINO_CKPT=${ROOT}/checkpoints/dinov3_vitb16_pretrain.pth
DINO_SOURCE=${ROOT}/dinov3
LOG_DIR=${ROOT}/logs/coverage_router

TRAIN_SHA=44f1b1e76da9e7acc2b44bd5e1d97b957bea756bd9e1c1c116457b7daa99e0e9
TEST_SHA=f83dcbceef9d4443e34b6544d161e02b5d437cece59231b6ba915567c1d562ab
DINO_SHA=73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c
DINO_SOURCE_FP=0c3cc6c9ee6167b66f2a53ef539056f2bedbd31039a9340c549a64b4b198400a
TRAIN_RECORDS=117727
TRAIN_PERSONS=113458
TEST_IMAGES=4782
TEST_PERSONS=4782
SEED=3106

DENSE_DIR=${ROOT}/experiments/coverage_router/gf_fulltrain_dense_seed3106
SUPPORT_DIR=${ROOT}/experiments/coverage_router/gf_fulltrain_support_k025_seed3106
K50_DIR=${ROOT}/experiments/coverage_router/gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106
MATCHED_DIR=${ROOT}/experiments/coverage_router/gf_fulltrain_dense_decoder_continue_seed3106
OUT_DIR=${ROOT}/experiments/coverage_router/gf_fulltrain_official_test_seed3106
PIPELINE_STARTED=${OUT_DIR}/pipeline_started.json
PIPELINE_COMPLETE=${OUT_DIR}/pipeline_complete.json
K50_RESULT=${OUT_DIR}/learned_k50_image_b16_fp32.json
MATCHED_RESULT=${OUT_DIR}/matched_dense_image_b16_fp32.json
COMPARISON_JSON=${OUT_DIR}/comparison.json
COMPARISON_MD=${OUT_DIR}/comparison.md
ARTIFACTS=(
  "${DENSE_DIR}/final.pt" "${SUPPORT_DIR}/final.pt" "${K50_DIR}/final.pt"
  "${MATCHED_DIR}/final.pt" "${K50_RESULT}" "${MATCHED_RESULT}"
  "${COMPARISON_JSON}" "${COMPARISON_MD}"
)

sha_file() { sha256sum "$1" | awk '{print $1}'; }

source_fingerprint() {
  (
    cd "${ROOT}"
    find dinov3 -type f -name '*.py' -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

clean_git() {
  local status current
  status=$(git -C "${ROOT}" status --porcelain --untracked-files=all)
  [[ -z "${status}" ]] || { echo "${status}" >&2; die "formal run requires a clean worktree"; }
  current=$(git -C "${ROOT}" rev-parse HEAD)
  [[ "${current}" == "${COMMIT}" ]] || die "runtime commit changed: ${current}"
}

check_hash() {
  local path=$1 expected=$2 label=$3 actual
  [[ -f "${path}" ]] || die "missing ${label}: ${path}"
  actual=$(sha_file "${path}")
  [[ "${actual}" == "${expected}" ]] || die "${label} SHA mismatch: ${actual}"
}

gpu_name() {
  nvidia-smi -i "$1" --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//'
}

idle_gpu() {
  local gpu=$1 pids
  nvidia-smi -i "${gpu}" --query-gpu=index,name --format=csv,noheader >/dev/null \
    || die "invalid physical GPU ${gpu}"
  pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "${pids}" ]] || die "GPU ${gpu} is busy with compute PID(s): ${pids}"
}

check_cuda() {
  CUDA_VISIBLE_DEVICES=$1 "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "exactly one visible CUDA device is required"
print("CUDA verified:", torch.cuda.get_device_name(0))
PY
}

assert_runtime() {
  clean_git
  check_hash "${TRAIN_JSON}" "${TRAIN_SHA}" "train annotation"
  check_hash "${DINO_CKPT}" "${DINO_SHA}" "DINOv3 checkpoint"
  [[ "$(source_fingerprint)" == "${DINO_SOURCE_FP}" ]] \
    || die "ignored DINOv3 Python source fingerprint changed"
  idle_gpu "$1"
}

marker_write() {
  local path=$1 kind=$2 name=$3 artifact1=${4:-none} artifact2=${5:-none}
  local sha1=none sha2=none
  [[ "${artifact1}" == none ]] || sha1=$(sha_file "${artifact1}")
  [[ "${artifact2}" == none ]] || sha2=$(sha_file "${artifact2}")
  "${PYTHON_BIN}" - "${path}" "${kind}" "${name}" "${COMMIT}" \
    "${DENSE_GPU}" "${ROUTED_GPU}" "${TRAIN_SHA}" "${DINO_SHA}" \
    "${DINO_SOURCE_FP}" "${TEST_SHA}" "${artifact1}" "${sha1}" \
    "${artifact2}" "${sha2}" <<'PY'
import datetime, json, sys
from pathlib import Path
p = Path(sys.argv[1])
payload = {
    "kind": sys.argv[2], "name": sys.argv[3], "runtime_git_commit": sys.argv[4],
    "dense_gpu": int(sys.argv[5]), "routed_gpu": int(sys.argv[6]),
    "train_annotation_sha256": sys.argv[7], "dinov3_checkpoint_sha256": sys.argv[8],
    "dinov3_source_fingerprint": sys.argv[9], "expected_test_annotation_sha256": sys.argv[10],
    "artifact1": sys.argv[11], "artifact1_sha256": sys.argv[12],
    "artifact2": sys.argv[13], "artifact2_sha256": sys.argv[14],
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with p.open("x", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True); f.write("\n")
PY
}

marker_check() {
  local path=$1 kind=$2 name=$3 artifact1=${4:-none} artifact2=${5:-none}
  local sha1=none sha2=none
  [[ "${artifact1}" == none ]] || sha1=$(sha_file "${artifact1}")
  [[ "${artifact2}" == none ]] || sha2=$(sha_file "${artifact2}")
  "${PYTHON_BIN}" - "${path}" "${kind}" "${name}" "${COMMIT}" \
    "${DENSE_GPU}" "${ROUTED_GPU}" "${TRAIN_SHA}" "${DINO_SHA}" \
    "${DINO_SOURCE_FP}" "${TEST_SHA}" "${artifact1}" "${sha1}" \
    "${artifact2}" "${sha2}" <<'PY'
import json, sys
from pathlib import Path
got = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
keys = ("kind", "name", "runtime_git_commit", "dense_gpu", "routed_gpu",
        "train_annotation_sha256", "dinov3_checkpoint_sha256",
        "dinov3_source_fingerprint", "expected_test_annotation_sha256",
        "artifact1", "artifact1_sha256", "artifact2", "artifact2_sha256")
values = [sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]),
          *sys.argv[7:15]]
for key, expected in zip(keys, values):
    if got.get(key) != expected:
        raise SystemExit(f"marker {sys.argv[1]} mismatch: {key}")
PY
}

validate_pipeline_complete() {
  "${PYTHON_BIN}" - "${PIPELINE_COMPLETE}" "${COMMIT}" "${ARTIFACTS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x["completed"] is True and x["runtime_git_commit"] == sys.argv[2]
for p in sys.argv[3:]: assert x["artifacts"][str(Path(p).resolve())] == sha(p)
PY
}

cd "${ROOT}"
clean_git
[[ -d "${DINO_SOURCE}" ]] || die "missing ignored DINOv3 source: ${DINO_SOURCE}"
command -v sha256sum >/dev/null || die "sha256sum is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
check_cuda "${DENSE_GPU}"
check_cuda "${ROUTED_GPU}"
idle_gpu "${DENSE_GPU}"
idle_gpu "${ROUTED_GPU}"
check_hash "${TRAIN_JSON}" "${TRAIN_SHA}" "train annotation"
check_hash "${DINO_CKPT}" "${DINO_SHA}" "DINOv3 checkpoint"
[[ "$(source_fingerprint)" == "${DINO_SOURCE_FP}" ]] || die "DINOv3 source fingerprint mismatch"

"${PYTHON_BIN}" - "${TRAIN_JSON}" "${TRAIN_RECORDS}" "${TRAIN_PERSONS}" <<'PY'
import json, sys
records = json.load(open(sys.argv[1], encoding="utf-8"))
persons = sum(h.get("inout", 1) == 1 for r in records for h in r.get("heads", []))
assert len(records) == int(sys.argv[2]), (len(records), sys.argv[2])
assert persons == int(sys.argv[3]), (persons, sys.argv[3])
print(f"TRAIN_DATA verified records={len(records)} persons={persons}")
PY

TESTS=(tests/test_coverage_router_training_protocol.py tests/test_gazefollow_dataset_views.py)
[[ ! -f tests/test_validate_coverage_router_refit.py ]] || TESTS+=(tests/test_validate_coverage_router_refit.py)
[[ ! -f tests/test_formal_coverage_router_eval.py ]] || TESTS+=(tests/test_formal_coverage_router_eval.py)
"${PYTHON_BIN}" -m pytest -q "${TESTS[@]}"
clean_git

mkdir -p "${LOG_DIR}"
if [[ -e "${OUT_DIR}" && ! -d "${OUT_DIR}" ]]; then die "OUT_DIR is not a directory"; fi
if [[ -d "${OUT_DIR}" && ! -f "${PIPELINE_STARTED}" ]] \
  && find "${OUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  die "non-empty OUT_DIR has no pipeline_started.json"
fi
mkdir -p "${OUT_DIR}"
if [[ -f "${PIPELINE_STARTED}" ]]; then
  marker_check "${PIPELINE_STARTED}" pipeline started
else
  marker_write "${PIPELINE_STARTED}" pipeline started
fi

stage_props() {
  case "$1" in
    dense)
      DIR=${DENSE_DIR}; EXP=gf_fulltrain_dense_seed3106; ROUTER_STAGE=support_pilot
      KEEP=1.0; EPOCHS=15; BATCH=60; ACCUM=1; WARMUP=0
      LR_R=0; LR_D=1e-3; HEAT=1; COV=0; BUDGET=0; R_COUNT=0; D_COUNT=3416576
      INIT=none; REINIT=false; CLIP=none
      ;;
    support)
      DIR=${SUPPORT_DIR}; EXP=gf_fulltrain_support_k025_seed3106; ROUTER_STAGE=support_pilot
      KEEP=0.25; EPOCHS=3; BATCH=16; ACCUM=1; WARMUP=0
      LR_R=1e-3; LR_D=0; HEAT=0; COV=1; BUDGET=0.05; R_COUNT=463105; D_COUNT=0
      INIT=${DENSE_DIR}/final.pt; REINIT=true; CLIP=none
      ;;
    k50)
      DIR=${K50_DIR}; EXP=gf_fulltrain_sparse_fixedrouter_decoder_k050_seed3106; ROUTER_STAGE=backbone_sparse
      KEEP=0.50; EPOCHS=5; BATCH=16; ACCUM=2; WARMUP=3
      LR_R=0; LR_D=1e-4; HEAT=1; COV=0; BUDGET=0; R_COUNT=0; D_COUNT=3416576
      INIT=${SUPPORT_DIR}/final.pt; REINIT=false; CLIP=1.0
      ;;
    matched)
      DIR=${MATCHED_DIR}; EXP=gf_fulltrain_dense_decoder_continue_seed3106; ROUTER_STAGE=support_pilot
      KEEP=1.0; EPOCHS=5; BATCH=16; ACCUM=2; WARMUP=0
      LR_R=0; LR_D=1e-4; HEAT=1; COV=0; BUDGET=0; R_COUNT=0; D_COUNT=3416576
      INIT=${DENSE_DIR}/final.pt; REINIT=false; CLIP=1.0
      ;;
    *) die "unknown stage $1" ;;
  esac
  FINAL=${DIR}/final.pt
  RESUME=${DIR}/last.resume.pt
  STAGE_MARKER=${OUT_DIR}/stage_$1.completed
  STAGE_LOG=${LOG_DIR}/${EXP}.log
}

validate_stage() {
  stage_props "$1"
  "${PYTHON_BIN}" scripts/validate_coverage_router_refit.py \
    --run_dir "${DIR}" --seed "${SEED}" --router_stage "${ROUTER_STAGE}" \
    --keep_ratio "${KEEP}" --epochs "${EPOCHS}" --batch_size "${BATCH}" \
    --router_trainable "${R_COUNT}" --decoder_trainable "${D_COUNT}" --inout_trainable 0 \
    --init_checkpoint "${INIT}" --git_commit "${COMMIT}" \
    --lr_router "${LR_R}" --lr_decoder "${LR_D}" \
    --router_warmup_epochs "${WARMUP}" --grad_accum_steps "${ACCUM}" \
    --reinitialize_router_on_init "${REINIT}" --checkpoint_role fixed_epoch_final
}

train_new() {
  local stage=$1 gpu=$2
  stage_props "${stage}"
  [[ "${INIT}" == none || -f "${INIT}" ]] || die "${stage} init checkpoint missing: ${INIT}"
  [[ ! -e "${STAGE_LOG}" ]] || die "new ${stage} stage has an existing log: ${STAGE_LOG}"
  : > "${STAGE_LOG}"
  local cmd=("${PYTHON_BIN}" -u scripts/train_coverage_router.py
    --formal_full_train_no_eval --dataset gazefollow --model gazelle_dinov3_vitb16
    --data_path "${DATA}" --router_stage "${ROUTER_STAGE}" --route_after_block 5
    --keep_ratio "${KEEP}" --router_hidden_dim 256 --router_temperature 1 --escape_tokens 8
    --spatial_prior none --fusion raw_concat --router_warmup_epochs "${WARMUP}"
    --heatmap_loss_weight "${HEAT}" --inout_loss_lambda 0
    --router_coverage_weight "${COV}" --router_budget_weight "${BUDGET}" --router_entropy_weight 0
    --lr_router "${LR_R}" --lr_decoder "${LR_D}" --lr_backbone 0 --lr_inout 0
    --weight_decay 0 --max_epochs "${EPOCHS}" --batch_size "${BATCH}"
    --grad_accum_steps "${ACCUM}" --n_workers 8 --log_iter 10
    --wandb_project GazeRoute --wandb_mode offline --exp_name "${EXP}"
    --run_dir "${DIR}" --seed "${SEED}")
  [[ "${INIT}" == none ]] || cmd+=(--init_ckpt "${INIT}")
  [[ "${REINIT}" != true ]] || cmd+=(--reinitialize_router_on_init)
  [[ "${CLIP}" == none ]] || cmd+=(--clip_grad_norm "${CLIP}")
  echo "FULLTRAIN_STAGE_START stage=${stage} mode=new gpu=${gpu} log=${STAGE_LOG}"
  CUDA_VISIBLE_DEVICES=${gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${cmd[@]}" >>"${STAGE_LOG}" 2>&1
}

train_resume() {
  local stage=$1 gpu=$2
  stage_props "${stage}"
  echo "FULLTRAIN_STAGE_START stage=${stage} mode=resume gpu=${gpu} log=${STAGE_LOG}"
  echo "RESUME $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=${COMMIT}" >>"${STAGE_LOG}"
  CUDA_VISIBLE_DEVICES=${gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" -u scripts/train_coverage_router.py \
      --resume "${RESUME}" --run_dir "${DIR}" --n_workers 8 --log_iter 10 \
      --wandb_project GazeRoute --wandb_mode offline --exp_name "${EXP}" \
      >>"${STAGE_LOG}" 2>&1
}

ensure_stage() {
  local stage=$1 gpu=$2
  stage_props "${stage}"
  if [[ -f "${STAGE_MARKER}" ]]; then
    [[ -f "${FINAL}" ]] || die "${stage} marker exists but final.pt is missing"
    validate_stage "${stage}"; marker_check "${STAGE_MARKER}" stage "${stage}" "${FINAL}"
    echo "FULLTRAIN_STAGE_COMPLETE stage=${stage} mode=validated_skip"
    return
  fi
  if [[ -f "${FINAL}" ]]; then
    echo "FULLTRAIN_STAGE_RECOVERY stage=${stage} action=validate_final"
    if validate_stage "${stage}"; then
      marker_write "${STAGE_MARKER}" stage "${stage}" "${FINAL}"
      echo "FULLTRAIN_STAGE_COMPLETE stage=${stage} epoch=$((EPOCHS - 1))"
      return
    fi
    if [[ ! -f "${DIR}/summary.json" && -f "${RESUME}" ]]; then
      echo "FULLTRAIN_STAGE_RECOVERY stage=${stage} action=complete_missing_summary_from_resume"
      assert_runtime "${gpu}"; train_resume "${stage}" "${gpu}"
    else
      die "${stage} final.pt failed validation for a reason other than a missing summary; do not overwrite it, audit manually"
    fi
  elif [[ -f "${RESUME}" ]]; then
    assert_runtime "${gpu}"; train_resume "${stage}" "${gpu}"
  elif [[ -e "${DIR}" ]]; then
    die "partial ${stage} has neither final.pt nor last.resume.pt: ${DIR}"
  else
    assert_runtime "${gpu}"; train_new "${stage}" "${gpu}"
  fi
  # Re-pin code, data, DINO weights/source, and GPU ownership after the long
  # training process returns, before accepting any artifact as formal.
  assert_runtime "${gpu}"
  validate_stage "${stage}"
  marker_write "${STAGE_MARKER}" stage "${stage}" "${FINAL}"
  echo "FULLTRAIN_STAGE_COMPLETE stage=${stage} epoch=$((EPOCHS - 1))"
}

# Dense base must finish before either branch is allowed to start.
ensure_stage dense "${DENSE_GPU}"

# Route learning/adaptation and matched-dense continuation are independent after dense15.
(
  ensure_stage support "${ROUTED_GPU}"
  ensure_stage k50 "${ROUTED_GPU}"
) &
ROUTED_PID=$!
(ensure_stage matched "${DENSE_GPU}") &
MATCHED_PID=$!
BRANCH_FAILED=0
if ! wait "${ROUTED_PID}"; then BRANCH_FAILED=1; fi
if ! wait "${MATCHED_PID}"; then BRANCH_FAILED=1; fi
[[ "${BRANCH_FAILED}" -eq 0 ]] || die "one or both full-train branches failed; inspect stage logs"

if [[ -f "${PIPELINE_COMPLETE}" ]]; then
  validate_pipeline_complete
  echo "GAZEFOLLOW_FULLTRAIN_PIPELINE complete mode=validated_skip"
  exit 0
fi

# The first official-test read is deliberately below all four training validators.
assert_runtime "${DENSE_GPU}"
idle_gpu "${ROUTED_GPU}"
for id in k50 dense; do
  started=${OUT_DIR}/official_test_${id}.started
  completed=${OUT_DIR}/official_test_${id}.completed
  if [[ -f "${started}" && ! -f "${completed}" ]]; then
    die "${id} has .started without .completed; refuse any automatic second official-test read and audit manually"
  fi
done
ACCESS_MARKER=${OUT_DIR}/official_test_access.started
if [[ -f "${ACCESS_MARKER}" ]]; then marker_check "${ACCESS_MARKER}" official access; else marker_write "${ACCESS_MARKER}" official access; fi
check_hash "${TEST_JSON}" "${TEST_SHA}" "official-test annotation"
"${PYTHON_BIN}" - "${TEST_JSON}" "${TEST_IMAGES}" "${TEST_PERSONS}" <<'PY'
import json, sys
records = json.load(open(sys.argv[1], encoding="utf-8"))
persons = sum(h.get("inout", 1) == 1 for r in records for h in r.get("heads", []))
assert len(records) == int(sys.argv[2]); assert persons == int(sys.argv[3])
print(f"OFFICIAL_TEST_DATA verified images={len(records)} persons={persons}")
PY

validate_eval() {
  local id=$1 checkpoint=$2 stage=$3 keep=$4 output=$5
  "${PYTHON_BIN}" scripts/validate_coverage_router_eval.py \
    --result "${output}" --checkpoint "${checkpoint}" --official_annotation "${TEST_JSON}" \
    --expect_dataset gazefollow --expect_gazefollow_eval_unit image --expect_fp32 \
    --expect_official_annotation_sha256 "${TEST_SHA}" \
    --expect_runtime_git_commit "${COMMIT}"
  "${PYTHON_BIN}" - "${output}" "${stage}" "${keep}" "${TEST_IMAGES}" "${TEST_PERSONS}" <<'PY'
import json, math, sys
r=json.load(open(sys.argv[1], encoding="utf-8")); c=r["evaluation_config"]; m=r["metrics"]
assert c["batch_size"]==16 and c["n_workers"]==8 and c["device"]=="cuda"
assert r["checkpoint_epoch"]==4 and r["model_config"]["router_stage"]==sys.argv[2]
assert math.isclose(float(r["model_config"]["keep_ratio"]), float(sys.argv[3]), abs_tol=1e-12)
assert r["dataset_image_count"]==int(sys.argv[4]) and r["dataset_sample_count"]==int(sys.argv[5])
for k in ("auc","avg_l2","min_l2","routing_actual_keep_ratio","routing_gt_point_coverage","routing_hard_coverage","routing_soft_coverage"):
    assert isinstance(m.get(k),(int,float)) and math.isfinite(float(m[k])), (k,m.get(k))
PY
}

run_eval_once() {
  local id=$1 display=$2 checkpoint=$3 stage=$4 keep=$5 output=$6 gpu=$7
  local started=${OUT_DIR}/official_test_${id}.started
  local completed=${OUT_DIR}/official_test_${id}.completed
  local log=${LOG_DIR}/gf_fulltrain_official_test_${id}_b16_fp32.log
  if [[ -f "${completed}" ]]; then
    [[ -f "${started}" && -f "${output}" ]] || die "${id} completed marker is inconsistent"
    validate_eval "${id}" "${checkpoint}" "${stage}" "${keep}" "${output}"
    marker_check "${started}" eval_started "${id}" "${checkpoint}"
    marker_check "${completed}" eval_completed "${id}" "${checkpoint}" "${output}"
    echo "OFFICIAL_TEST_COMPLETE model=${display} mode=validated_skip"; return
  fi
  [[ ! -f "${started}" ]] || die "${id} has .started without .completed; refuse automatic re-evaluation and audit manually"
  [[ ! -e "${output}" && ! -e "${log}" ]] || die "${id} has an unmarked result/log; refusing overwrite"
  assert_runtime "${gpu}"
  marker_write "${started}" eval_started "${id}" "${checkpoint}"
  : >"${log}"
  echo "OFFICIAL_TEST_START model=${display} gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" -u scripts/eval_coverage_router.py \
    --checkpoint "${checkpoint}" --dataset gazefollow --data_path "${DATA}" \
    --gazefollow_eval_split official_test --gazefollow_eval_unit image \
    --gazefollow_head_count_subset all --batch_size 16 --n_workers 8 --device cuda \
    --require_full_train_no_eval --output "${output}" >>"${log}" 2>&1
  validate_eval "${id}" "${checkpoint}" "${stage}" "${keep}" "${output}"
  marker_write "${completed}" eval_completed "${id}" "${checkpoint}" "${output}"
  echo "OFFICIAL_TEST_COMPLETE model=${display}"
}

(run_eval_once k50 learned_k50 "${K50_DIR}/final.pt" backbone_sparse 0.50 "${K50_RESULT}" "${ROUTED_GPU}") &
K50_EVAL_PID=$!
(run_eval_once dense matched_dense "${MATCHED_DIR}/final.pt" support_pilot 1.0 "${MATCHED_RESULT}" "${DENSE_GPU}") &
DENSE_EVAL_PID=$!
EVAL_FAILED=0
if ! wait "${K50_EVAL_PID}"; then EVAL_FAILED=1; fi
if ! wait "${DENSE_EVAL_PID}"; then EVAL_FAILED=1; fi
[[ "${EVAL_FAILED}" -eq 0 ]] || die "one or both official evaluations failed; .started markers forbid automatic retry"

"${PYTHON_BIN}" - "${K50_RESULT}" "${MATCHED_RESULT}" "${COMPARISON_JSON}" "${COMPARISON_MD}" <<'PY'
import json, os, sys
from pathlib import Path
k=json.load(open(sys.argv[1],encoding="utf-8")); d=json.load(open(sys.argv[2],encoding="utf-8"))
names=("auc","avg_l2","min_l2","routing_actual_keep_ratio","routing_gt_point_coverage","routing_hard_coverage","routing_soft_coverage")
km={n:k["metrics"][n] for n in names}; dm={n:d["metrics"][n] for n in names}
out={"policy":"fixed_checkpoints_no_post_test_selection","accuracy_gate":None,"learned_k50":km,"matched_dense":dm,"k50_minus_dense":{n:km[n]-dm[n] for n in names}}
j=Path(sys.argv[3]); t=j.with_suffix(".json.tmp"); t.write_text(json.dumps(out,allow_nan=False,indent=2,sort_keys=True)+"\n"); os.replace(t,j)
rows=["# GazeFollow official-test comparison","","Fixed checkpoints; no post-test selection.","","| metric | K50 | dense | delta |","|---|---:|---:|---:|"]
rows += [f"| {n} | {km[n]:.6f} | {dm[n]:.6f} | {km[n]-dm[n]:+.6f} |" for n in names]
m=Path(sys.argv[4]); t=m.with_suffix(".md.tmp"); t.write_text("\n".join(rows)+"\n"); os.replace(t,m)
PY

"${PYTHON_BIN}" - "${PIPELINE_COMPLETE}" "${COMMIT}" "${DENSE_GPU}" "${ROUTED_GPU}" \
  "${TRAIN_SHA}" "${TEST_SHA}" "${DINO_SHA}" "${DINO_SOURCE_FP}" "${ARTIFACTS[@]}" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=Path(sys.argv[1]); paths=[Path(x).resolve() for x in sys.argv[9:]]
x={"completed":True,"protocol":"gazefollow_fulltrain_two_gpu_v1","runtime_git_commit":sys.argv[2],
   "dense_gpu":int(sys.argv[3]),"routed_gpu":int(sys.argv[4]),"train_annotation_sha256":sys.argv[5],
   "official_test_annotation_sha256":sys.argv[6],"dinov3_checkpoint_sha256":sys.argv[7],
   "dinov3_source_fingerprint":sys.argv[8],"official_test_policy":"exactly_once_per_fixed_checkpoint",
   "artifacts":{str(a):sha(a) for a in paths},"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
with p.open("x",encoding="utf-8") as f: json.dump(x,f,indent=2,sort_keys=True); f.write("\n")
PY

validate_pipeline_complete
echo "GAZEFOLLOW_FULLTRAIN_PIPELINE complete"
