#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

EXPERIMENT_TAG=${EXPERIMENT_TAG:?Set EXPERIMENT_TAG}
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
PROCESSED_ROOT=${PROCESSED_ROOT:?Set PROCESSED_ROOT}
TRAIN_CSV=${TRAIN_CSV:?Set TRAIN_CSV}
VAL_CSV=${VAL_CSV:?Set VAL_CSV}
PREPROCESS_SUMMARY=${PREPROCESS_SUMMARY:?Set PREPROCESS_SUMMARY}
ROI_INDEX=${ROI_INDEX:?Set ROI_INDEX}
LPIPS_STATE=${LPIPS_STATE:?Set LPIPS_STATE}
INPUT_SIZE=${INPUT_SIZE:?Set INPUT_SIZE}
GLOBAL_CROP_SIZE=${GLOBAL_CROP_SIZE:?Set GLOBAL_CROP_SIZE}
ROI_CROP_SIZE=${ROI_CROP_SIZE:?Set ROI_CROP_SIZE}
EXPECTED_SPACING=${EXPECTED_SPACING:?Set EXPECTED_SPACING as 'sx sy sz'}
ROI_POSITIVE_SAMPLE_COUNTS=${ROI_POSITIVE_SAMPLE_COUNTS:?Set eight crop-specific counts}
ROI_FREQUENCY_DATASET_SIZE=${ROI_FREQUENCY_DATASET_SIZE:?Set training manifest size}

FUSION_MODE=${FUSION_MODE:-linear_sqrt}
LAMBDA_SEG=${LAMBDA_SEG:-0.0}
POSITIVE_ROI_LOSS_WEIGHT=${POSITIVE_ROI_LOSS_WEIGHT:-0.25}
FUSION_REFERENCE_COUNT=${FUSION_REFERENCE_COUNT:-9}
ROI_FREQUENCY_ALPHA=${ROI_FREQUENCY_ALPHA:-0.5}
ROI_MAX_WEIGHT_RATIO=${ROI_MAX_WEIGHT_RATIO:-4.0}
ORGAN_ROI_PROBABILITY=${ORGAN_ROI_PROBABILITY:-0.2}
N_GPU=${N_GPU:-2}
MICRO_BATCH=${MICRO_BATCH:-8}
ACCUM_ITER=${ACCUM_ITER:-12}
TARGET_EFFECTIVE_BATCH=${TARGET_EFFECTIVE_BATCH:-192}
MAX_UPDATES=${MAX_UPDATES:-118800}
WARMUP_UPDATES=${WARMUP_UPDATES:-5940}
BASE_LR=${BASE_LR:-1e-4}
WORKERS=${WORKERS:-10}
SAVE_FREQ=${SAVE_FREQ:-100}
MASTER_PORT=${MASTER_PORT:-25741}
MAX_STEPS_PER_EPOCH=${MAX_STEPS_PER_EPOCH:-}
NO_SAVE=${NO_SAVE:-0}
GPU_IDS=${GPU_IDS:-}

if (( INPUT_SIZE <= 0 || INPUT_SIZE % 16 != 0 )); then
  echo "INPUT_SIZE must be positive and divisible by 16" >&2
  exit 2
fi
if (( ROI_CROP_SIZE > GLOBAL_CROP_SIZE )); then
  echo "ROI_CROP_SIZE must not exceed GLOBAL_CROP_SIZE" >&2
  exit 2
fi
EFFECTIVE_BATCH=$((MICRO_BATCH * ACCUM_ITER * N_GPU))
if [[ "${EFFECTIVE_BATCH}" -ne "${TARGET_EFFECTIVE_BATCH}" ]]; then
  echo "Effective batch ${EFFECTIVE_BATCH} != required ${TARGET_EFFECTIVE_BATCH}" >&2
  exit 2
fi
for required in "${PYTHON}" "${TRAIN_CSV}" "${VAL_CSV}" "${PREPROCESS_SUMMARY}" "${ROI_INDEX}" "${LPIPS_STATE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
read -r -a SPACING_ARRAY <<< "${EXPECTED_SPACING}"
read -r -a ROI_COUNTS_ARRAY <<< "${ROI_POSITIVE_SAMPLE_COUNTS}"
if [[ "${#SPACING_ARRAY[@]}" -ne 3 || "${#ROI_COUNTS_ARRAY[@]}" -ne 8 ]]; then
  echo "EXPECTED_SPACING needs 3 values and ROI_POSITIVE_SAMPLE_COUNTS needs 8" >&2
  exit 2
fi

RUN_NAME=${RUN_NAME:-OrgSlot_${EXPERIMENT_TAG}_Loss3_ROI20_eb${TARGET_EFFECTIVE_BATCH}_u${MAX_UPDATES}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}}
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi
export OMP_NUM_THREADS=1
"${PYTHON}" -c "import os, torch; expected=int('${N_GPU}'); count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch_cuda=', torch.__version__, torch.version.cuda, 'count=', count); assert torch.cuda.is_available() and count == expected; torch.ones(1, device='cuda')"

COMMAND=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${N_GPU}"
  --master_port="${MASTER_PORT}"
  main_pretrain_orgslot_common8_a100.py
  --batch_size "${MICRO_BATCH}"
  --accum_iter "${ACCUM_ITER}"
  --input_size "${INPUT_SIZE}"
  --global_crop_size "${GLOBAL_CROP_SIZE}"
  --roi_crop_size "${ROI_CROP_SIZE}"
  --expected_spacing "${SPACING_ARRAY[@]}"
  --roi_center_jitter 0.1
  --focus_class_ids 4,5,6
  --organ_roi_aug
  --organ_roi_probability "${ORGAN_ROI_PROBABILITY}"
  --roi_index "${ROI_INDEX}"
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
  --fusion_reference_count "${FUSION_REFERENCE_COUNT}"
  --lambda_seg "${LAMBDA_SEG}"
  --positive_roi_loss_weight "${POSITIVE_ROI_LOSS_WEIGHT}"
  --roi_positive_sample_counts "${ROI_COUNTS_ARRAY[@]}"
  --roi_frequency_dataset_size "${ROI_FREQUENCY_DATASET_SIZE}"
  --roi_frequency_alpha "${ROI_FREQUENCY_ALPHA}"
  --roi_max_weight_ratio "${ROI_MAX_WEIGHT_RATIO}"
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
if [[ -n "${MAX_STEPS_PER_EPOCH}" ]]; then
  COMMAND+=(--max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}")
fi
if [[ "${NO_SAVE}" == "1" ]]; then
  COMMAND+=(--no_save)
fi

{
  echo "experiment_tag=${EXPERIMENT_TAG}"
  echo "processed_root=${PROCESSED_ROOT}"
  echo "train_csv=${TRAIN_CSV}"
  echo "spacing=${EXPECTED_SPACING}"
  echo "input=${INPUT_SIZE} global_crop=${GLOBAL_CROP_SIZE} roi_crop=${ROI_CROP_SIZE}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "fusion_mode=${FUSION_MODE} fusion_reference_count=${FUSION_REFERENCE_COUNT} lambda_seg=${LAMBDA_SEG}"
  echo "positive_roi_loss_weight=${POSITIVE_ROI_LOSS_WEIGHT}"
  echo "roi_counts=${ROI_POSITIVE_SAMPLE_COUNTS}"
  echo "micro_batch=${MICRO_BATCH} accum_iter=${ACCUM_ITER} effective_batch=${EFFECTIVE_BATCH}"
  echo "max_updates=${MAX_UPDATES} warmup_updates=${WARMUP_UPDATES}"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} | tee "${OUTPUT_DIR}/launcher.log"

cd "${REPO_ROOT}"
"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
