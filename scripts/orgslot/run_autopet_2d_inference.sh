#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

METHOD=${1:?Usage: $0 METHOD GPU_ID RUN_NAME [MAX_SAMPLES]}
GPU_ID=${2:?Usage: $0 METHOD GPU_ID RUN_NAME [MAX_SAMPLES]}
RUN_NAME=${3:?Usage: $0 METHOD GPU_ID RUN_NAME [MAX_SAMPLES]}
MAX_SAMPLES=${4:-}
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
TEST_CSV=${TEST_CSV:-/mnt/DATA-4/anteng/Test_A100_Final112.csv}
BATCH_SIZE=${BATCH_SIZE:-64}
WORKERS=${WORKERS:-8}
THRESHOLD=${THRESHOLD:-0.02}
FUSION_MODE=${FUSION_MODE:-post_layernorm}
EVAL_ROOT=${EVAL_ROOT:-${REPO_ROOT}/Results/OrganSlotBank/evaluation/AbdAutoPet_2D/reconstruction_threshold_v1}
OWT_CHECKPOINT=${OWT_CHECKPOINT:-/mnt/DATA-4/sifan2/Scripts/OWT/Results/AbdAutoPet_2D/Token_mae_vit_base_patch16-LA_1e-4_4_224_20_v11_v01_L2-LPIPS_GPU2_96_1200/checkpoint-1199.pth}
ORGSLOT_CHECKPOINT=${ORGSLOT_CHECKPOINT:-${REPO_ROOT}/Results/OrganSlotBank/OWTLegacy/AbdAutoPet_2D/OrgSlot_mae_vit_base_patch16-LA_1e-4_4_224_20_v11_v01_L2-LPIPS_GPU2_96_1200_legacy-mask/checkpoint-1199.pth}

case "${METHOD}" in
  owt) CHECKPOINT=${OWT_CHECKPOINT} ;;
  orgslot) CHECKPOINT=${ORGSLOT_CHECKPOINT} ;;
  *) echo "METHOD must be owt or orgslot" >&2; exit 2 ;;
esac

if [[ ! -x "${PYTHON}" || ! -f "${TEST_CSV}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Missing Python, test CSV, or checkpoint" >&2
  exit 2
fi

OUTPUT_DIR="${EVAL_ROOT}/${RUN_NAME}_${METHOD}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 2
fi

COMMAND=(
  "${PYTHON}" -m tools.eval_autopet_reconstruction_threshold
  --method "${METHOD}"
  --checkpoint "${CHECKPOINT}"
  --test-csv "${TEST_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --threshold "${THRESHOLD}"
  --min-size 20
  --opening-radius 1
  --class-ids 0,1,2,3,4
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
printf 'command=' > "${EVAL_ROOT}/${RUN_NAME}_${METHOD}.command.txt"
printf '%q ' "${COMMAND[@]}" >> "${EVAL_ROOT}/${RUN_NAME}_${METHOD}.command.txt"
printf '\n' >> "${EVAL_ROOT}/${RUN_NAME}_${METHOD}.command.txt"
"${COMMAND[@]}" 2>&1 | tee "${EVAL_ROOT}/${RUN_NAME}_${METHOD}.log"
