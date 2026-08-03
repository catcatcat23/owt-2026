#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
TRAIN_CSV=${TRAIN_CSV:-/mnt/DATA-4/anteng/Training_A100_Final112.csv}
GPU_IDS=${GPU_IDS:-0,1}
N_GPU=${N_GPU:-2}
MASTER_PORT=${MASTER_PORT:-25667}
EPOCHS=${EPOCHS:-1200}
BATCH_SIZE=${BATCH_SIZE:-96}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-60}
BASE_LR=${BASE_LR:-1e-4}
TOKEN_FACTOR=${TOKEN_FACTOR:-20}
FUSION_MODE=${FUSION_MODE:-post_layernorm}
LAMBDA_SEG=${LAMBDA_SEG:-0.0}
SAVE_FREQ=${SAVE_FREQ:-100}
WORKERS=${WORKERS:-10}
LPIPS_STATE=${LPIPS_STATE:-/mnt/DATA-4/anteng/pretrained/owt_lpips_vgg16.pth}
MAX_STEPS_PER_EPOCH=${MAX_STEPS_PER_EPOCH:-}
NO_SAVE=${NO_SAVE:-0}
RUN_NAME=${RUN_NAME:-OrgSlot_mae_vit_base_patch16-LA_1e-4_4_224_20_v11_v01_L2-LPIPS_GPU2_96_1200_legacy-mask}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/Results/OrganSlotBank/OWTLegacy/AbdAutoPet_2D/${RUN_NAME}}

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "Training manifest not found: ${TRAIN_CSV}" >&2
  exit 2
fi
if [[ ! -f "${LPIPS_STATE}" ]]; then
  echo "Local LPIPS state not found: ${LPIPS_STATE}" >&2
  echo "Extract it once from the matched original OWT checkpoint before DDP." >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}/checkpoint-1199.pth" ]]; then
  echo "Final checkpoint already exists; refusing to retrain: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

COMMAND=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${N_GPU}"
  --master_port="${MASTER_PORT}"
  main_pretrain_orgslot_a100.py
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --blr "${BASE_LR}"
  --weight_decay 0.05
  --input_size 224
  --token_factor "${TOKEN_FACTOR}"
  --slot_tg_depth 1
  --fusion_mode "${FUSION_MODE}"
  --loss_version L2-LPIPS
  --lambda_lpips 1.0
  --lpips_state "${LPIPS_STATE}"
  --lambda_seg "${LAMBDA_SEG}"
  --tgr_mode legacy_batch
  --data_path "${TRAIN_CSV}"
  --visibility_config configs/orgslot/autopet_all4_offline.json
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${OUTPUT_DIR}"
  --save_freq "${SAVE_FREQ}"
  --num_workers "${WORKERS}"
  --print_freq 20
  --seed 0
)

if [[ -n "${MAX_STEPS_PER_EPOCH}" ]]; then
  COMMAND+=(--max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}")
fi
if [[ "${NO_SAVE}" == "1" ]]; then
  COMMAND+=(--no_save)
fi

{
  echo "repo_root=${REPO_ROOT}"
  echo "train_csv=${TRAIN_CSV}"
  echo "gpu_ids=${GPU_IDS}"
  echo "n_gpu=${N_GPU}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "epochs=${EPOCHS} batch_per_gpu=${BATCH_SIZE} warmup=${WARMUP_EPOCHS}"
  echo "loss=L2+LPIPS lambda_seg=${LAMBDA_SEG} tgr_mode=legacy_batch fusion_mode=${FUSION_MODE}"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} | tee -a "${OUTPUT_DIR}/launcher.log"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=1
"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
