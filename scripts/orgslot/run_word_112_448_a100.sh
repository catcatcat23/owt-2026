#!/usr/bin/env bash
set -euo pipefail

ARM=${1:?Usage: $0 control|roi20}
case "${ARM}" in
  control|roi20) ;;
  *) echo "ARM must be control or roi20" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
DATA_ROOT=${DATA_ROOT:-/mnt/DATA-4/anteng}
PROCESSED_ROOT=${PROCESSED_ROOT:-${DATA_ROOT}/processed/OWT_Common8_112_NATIVE/WORD}
TRAIN_CSV=${TRAIN_CSV:-${PROCESSED_ROOT}/csv/WORD_Training_2D_native112.csv}
VAL_CSV=${VAL_CSV:-${PROCESSED_ROOT}/csv/WORD_Test_2D_native112.csv}
PREPROCESS_SUMMARY=${PREPROCESS_SUMMARY:-${PROCESSED_ROOT}/metadata/preprocess_summary.json}
ROI_INDEX=${ROI_INDEX:-${PROCESSED_ROOT}/metadata/small_organ_roi_index.json}
LPIPS_STATE=${LPIPS_STATE:-${DATA_ROOT}/pretrained/owt_lpips_vgg16.pth}

FUSION_MODE=${FUSION_MODE:?Set FUSION_MODE after the current AutoPET evaluations}
LAMBDA_SEG=${LAMBDA_SEG:?Set LAMBDA_SEG after the current AutoPET evaluations}
GPU_IDS=${GPU_IDS:-0,1}
N_GPU=${N_GPU:-2}
MASTER_PORT=${MASTER_PORT:-25741}
MICRO_BATCH=${MICRO_BATCH:-8}
ACCUM_ITER=${ACCUM_ITER:-12}
TARGET_EFFECTIVE_BATCH=${TARGET_EFFECTIVE_BATCH:-192}
MAX_UPDATES=${MAX_UPDATES:-118800}
WARMUP_UPDATES=${WARMUP_UPDATES:-5940}
BASE_LR=${BASE_LR:-1e-4}
WORKERS=${WORKERS:-10}
SAVE_FREQ=${SAVE_FREQ:-100}
MAX_STEPS_PER_EPOCH=${MAX_STEPS_PER_EPOCH:-}
NO_SAVE=${NO_SAVE:-0}

EFFECTIVE_BATCH=$((MICRO_BATCH * ACCUM_ITER * N_GPU))
if [[ "${EFFECTIVE_BATCH}" -ne "${TARGET_EFFECTIVE_BATCH}" ]]; then
  echo "Effective batch ${EFFECTIVE_BATCH} != required ${TARGET_EFFECTIVE_BATCH}" >&2
  exit 2
fi
for required in "${PYTHON}" "${TRAIN_CSV}" "${VAL_CSV}" "${PREPROCESS_SUMMARY}" "${LPIPS_STATE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ "${ARM}" == "roi20" && ! -f "${ROI_INDEX}" ]]; then
  echo "Missing ROI index: ${ROI_INDEX}" >&2
  exit 2
fi

RUN_NAME=${RUN_NAME:-OrgSlot_WORD112_input448_${ARM}_${FUSION_MODE}_seg${LAMBDA_SEG}_eb${TARGET_EFFECTIVE_BATCH}_u${MAX_UPDATES}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}}
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

COMMAND=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${N_GPU}"
  --master_port="${MASTER_PORT}"
  main_pretrain_orgslot_common8_a100.py
  --batch_size "${MICRO_BATCH}"
  --accum_iter "${ACCUM_ITER}"
  --input_size 448
  --global_crop_size 448
  --roi_crop_size 384
  --roi_center_jitter 0.1
  --focus_class_ids 4,5,6
  --max_optimizer_updates "${MAX_UPDATES}"
  --warmup_updates "${WARMUP_UPDATES}"
  --blr "${BASE_LR}"
  --weight_decay 0.05
  --token_factor 20
  --slot_tg_depth 1
  --fusion_mode "${FUSION_MODE}"
  --loss_version L2-LPIPS
  --lambda_lpips 1.0
  --lpips_state "${LPIPS_STATE}"
  --lambda_seg "${LAMBDA_SEG}"
  --tgr_mode legacy_batch
  --data_path "${TRAIN_CSV}"
  --val_data_path "${VAL_CSV}"
  --preprocess_summary "${PREPROCESS_SUMMARY}"
  --visibility_config configs/orgslot/common8_offline.json
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${OUTPUT_DIR}"
  --save_freq "${SAVE_FREQ}"
  --num_workers "${WORKERS}"
  --print_freq 20
  --seed 0
)
if [[ "${ARM}" == "roi20" ]]; then
  COMMAND+=(
    --organ_roi_aug
    --organ_roi_probability 0.2
    --roi_index "${ROI_INDEX}"
  )
fi
if [[ -n "${MAX_STEPS_PER_EPOCH}" ]]; then
  COMMAND+=(--max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}")
fi
if [[ "${NO_SAVE}" == "1" ]]; then
  COMMAND+=(--no_save)
fi

{
  echo "arm=${ARM}"
  echo "data_root=${DATA_ROOT}"
  echo "train_csv=${TRAIN_CSV}"
  echo "gpu_ids=${GPU_IDS}"
  echo "fusion_mode=${FUSION_MODE} lambda_seg=${LAMBDA_SEG}"
  echo "micro_batch=${MICRO_BATCH} accum_iter=${ACCUM_ITER} effective_batch=${EFFECTIVE_BATCH}"
  echo "max_updates=${MAX_UPDATES} warmup_updates=${WARMUP_UPDATES}"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} | tee "${OUTPUT_DIR}/launcher.log"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=1
"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
