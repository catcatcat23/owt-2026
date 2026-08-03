#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

DIMENSION=${1:-2D}
GPU_ID=${2:-0}
DATA_ROOT=${DATA_ROOT:-/mnt/DATA-4/anteng}
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/Results/OrganSlotBank/_smoke}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-8}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-4}
MAX_STEPS=${MAX_STEPS:-1}
WORKERS=${WORKERS:-2}
LR=${LR:-1e-4}

case "${DIMENSION}" in
  2D)
    SOURCE_CSV="${DATA_ROOT}/Training_A100_Final112.csv"
    RUN_NAME="gate_b_2d_base"
    ;;
  3D)
    SOURCE_CSV="${DATA_ROOT}/Training_Fixfr4_A100_Final112.csv"
    RUN_NAME="gate_b_3d_fixfr4_base"
    ;;
  *)
    echo "DIMENSION must be 2D or 3D, got ${DIMENSION}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found or not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CSV}" ]]; then
  echo "Expected manifest is missing: ${SOURCE_CSV}" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}/zip" ]]; then
  echo "Expected image tree is missing: ${DATA_ROOT}/zip" >&2
  exit 2
fi

RUN_STAMP=$(date +%Y%m%d-%H%M%S)
RESULT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_${RUN_STAMP}"
MANIFEST_DIR="${RESULT_DIR}/manifests"
mkdir -p "${MANIFEST_DIR}"

exec > >(tee -a "${RESULT_DIR}/launcher.log") 2>&1

echo "repo_root=${REPO_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "dimension=${DIMENSION}"
echo "gpu_id=${GPU_ID}"
echo "result_dir=${RESULT_DIR}"

cd "${REPO_ROOT}"
"${PYTHON}" tools/split_orgslot_manifest.py \
  --source "${SOURCE_CSV}" \
  --output-dir "${MANIFEST_DIR}" \
  --validation-fraction 0.1 \
  --seed 0

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=8

nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
"${PYTHON}" -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

"${PYTHON}" main_pretrain_orgslot.py \
  --stage base \
  --data-path "${MANIFEST_DIR}/train.csv" \
  --val-data-path "${MANIFEST_DIR}/validation.csv" \
  --output-dir "${RESULT_DIR}" \
  --dimension "${DIMENSION}" \
  --model-size base \
  --slot-tg-depth 1 \
  --loss-version L2 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-val-samples "${MAX_VAL_SAMPLES}" \
  --max-steps "${MAX_STEPS}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --lambda-lpips 0 \
  --device cuda

echo "Smoke completed: ${RESULT_DIR}"
