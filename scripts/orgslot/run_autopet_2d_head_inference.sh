#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

GPU_ID=${1:?Usage: $0 GPU_ID RUN_NAME [MAX_SAMPLES]}
RUN_NAME=${2:?Usage: $0 GPU_ID RUN_NAME [MAX_SAMPLES]}
MAX_SAMPLES=${3:-}
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
TEST_CSV=${TEST_CSV:-/mnt/DATA-4/anteng/Test_A100_Final112.csv}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to a trained OrganSlot head checkpoint}
FUSION_MODE=${FUSION_MODE:-post_layernorm}
BATCH_SIZE=${BATCH_SIZE:-64}
WORKERS=${WORKERS:-8}
EVAL_ROOT=${EVAL_ROOT:-${REPO_ROOT}/Results/OrganSlotBank/evaluation/AbdAutoPet_2D/head_v1}
OUTPUT_DIR="${EVAL_ROOT}/${RUN_NAME}"

if [[ ! -x "${PYTHON}" || ! -f "${TEST_CSV}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Missing Python, test CSV, or checkpoint" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 2
fi

COMMAND=(
  "${PYTHON}" -m tools.eval_autopet_orgslot_heads
  --checkpoint "${CHECKPOINT}"
  --test-csv "${TEST_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --binary-threshold 0.5
  --min-size 20
  --opening-radius 1
  --print-freq 20
  --device cuda
  --fusion-mode "${FUSION_MODE}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMAND+=(--max-samples "${MAX_SAMPLES}")
fi

mkdir -p "${EVAL_ROOT}"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=4
printf 'command=' > "${EVAL_ROOT}/${RUN_NAME}.command.txt"
printf '%q ' "${COMMAND[@]}" >> "${EVAL_ROOT}/${RUN_NAME}.command.txt"
printf '
' >> "${EVAL_ROOT}/${RUN_NAME}.command.txt"
"${COMMAND[@]}" 2>&1 | tee "${EVAL_ROOT}/${RUN_NAME}.log"
