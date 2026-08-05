#!/usr/bin/env bash
set -euo pipefail

# Formal AB/BA efficiency sweep for the fixed split-clean K50 checkpoint.
# Run from the repository root with one explicitly selected CUDA device:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_coverage_router_efficiency_sweep.sh ...

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 PYTHON_BIN CHECKPOINT_PATH OUTPUT_DIR" >&2
  exit 2
fi

EFFICIENCY_PYTHON_BIN=$1
EFFICIENCY_CHECKPOINT_PATH=$2
EFFICIENCY_OUTPUT_DIR=$3

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one physical GPU index." >&2
  exit 2
fi

if [[ ! -x "${EFFICIENCY_PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${EFFICIENCY_PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${EFFICIENCY_CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${EFFICIENCY_CHECKPOINT_PATH}" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: the worktree is not clean; commit the runtime code before the formal sweep." >&2
  git status --short >&2
  exit 2
fi

mkdir -p "${EFFICIENCY_OUTPUT_DIR}"

echo "runtime_git_commit=$(git rev-parse HEAD)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,pstate --format=csv
nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv

for EFFICIENCY_BATCH_SIZE in 1 4 8; do
  echo "FORMAL_EFFICIENCY batch=${EFFICIENCY_BATCH_SIZE} order=forward"
  "${EFFICIENCY_PYTHON_BIN}" -u scripts/benchmark_coverage_router.py \
    --checkpoint "${EFFICIENCY_CHECKPOINT_PATH}" \
    --variants dense support k100 k50 \
    --device cuda \
    --batch_size "${EFFICIENCY_BATCH_SIZE}" \
    --num_people 1 \
    --image_size 512 \
    --warmup_iters 50 \
    --latency_iters 200 \
    --throughput_iters 500 \
    --repeats 5 \
    --amp \
    --output "${EFFICIENCY_OUTPUT_DIR}/benchmark_formal_b${EFFICIENCY_BATCH_SIZE}_forward.json"

  echo "FORMAL_EFFICIENCY batch=${EFFICIENCY_BATCH_SIZE} order=reverse"
  "${EFFICIENCY_PYTHON_BIN}" -u scripts/benchmark_coverage_router.py \
    --checkpoint "${EFFICIENCY_CHECKPOINT_PATH}" \
    --variants k50 k100 support dense \
    --device cuda \
    --batch_size "${EFFICIENCY_BATCH_SIZE}" \
    --num_people 1 \
    --image_size 512 \
    --warmup_iters 50 \
    --latency_iters 200 \
    --throughput_iters 500 \
    --repeats 5 \
    --amp \
    --output "${EFFICIENCY_OUTPUT_DIR}/benchmark_formal_b${EFFICIENCY_BATCH_SIZE}_reverse.json"

  nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,pstate --format=csv
done

echo "FORMAL_EFFICIENCY complete"
