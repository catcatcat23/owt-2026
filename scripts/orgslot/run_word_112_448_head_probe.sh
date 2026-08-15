#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
PROCESSED_ROOT=${PROCESSED_ROOT:-/mnt/DATA-4/anteng/processed/OWT_Common8_112_NATIVE/WORD}
TRAIN_CSV=${TRAIN_CSV:-${PROCESSED_ROOT}/csv/WORD_Training_2D_native112.csv}
VAL_CSV=${VAL_CSV:-${PROCESSED_ROOT}/csv/WORD_Test_2D_native112.csv}
PREPROCESS_SUMMARY=${PREPROCESS_SUMMARY:-${PROCESSED_ROOT}/metadata/preprocess_summary.json}
ROI_INDEX=${ROI_INDEX:-${PROCESSED_ROOT}/metadata/small_organ_roi_index.json}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-${REPO_ROOT}/Results/OrganSlotBank/Common8/WORD_2D/OrgSlot_WORD112_input448_roi20_linear_sqrt_seg0_eb192_u118800/checkpoint-802.pth}

GPU_IDS=${GPU_IDS:-0,1}
N_GPU=${N_GPU:-2}
MASTER_PORT=${MASTER_PORT:-25743}
MICRO_BATCH=${MICRO_BATCH:-8}
ACCUM_ITER=${ACCUM_ITER:-12}
TARGET_EFFECTIVE_BATCH=${TARGET_EFFECTIVE_BATCH:-192}
MAX_UPDATES=${MAX_UPDATES:-14800}
WARMUP_UPDATES=${WARMUP_UPDATES:-740}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
WORKERS=${WORKERS:-10}
SAVE_FREQ=${SAVE_FREQ:-20}

EFFECTIVE_BATCH=$((MICRO_BATCH * ACCUM_ITER * N_GPU))
if [[ "${EFFECTIVE_BATCH}" -ne "${TARGET_EFFECTIVE_BATCH}" ]]; then
  echo "Effective batch ${EFFECTIVE_BATCH} != required ${TARGET_EFFECTIVE_BATCH}" >&2
  exit 2
fi
for required in "${PYTHON}" "${TRAIN_CSV}" "${VAL_CSV}" "${PREPROCESS_SUMMARY}" "${ROI_INDEX}" "${INIT_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

RUN_NAME=${RUN_NAME:-OrgSlot_WORD112_input448_roi20_headonly_DiceBCE_from_roi20ckpt802_eb${TARGET_EFFECTIVE_BATCH}_u${MAX_UPDATES}}
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
  --training_scope head_only
  --init_checkpoint "${INIT_CHECKPOINT}"
  --batch_size "${MICRO_BATCH}"
  --accum_iter "${ACCUM_ITER}"
  --input_size 448
  --global_crop_size 448
  --roi_crop_size 384
  --roi_center_jitter 0.1
  --focus_class_ids 4,5,6
  --organ_roi_aug
  --organ_roi_probability 0.2
  --roi_index "${ROI_INDEX}"
  --max_optimizer_updates "${MAX_UPDATES}"
  --warmup_updates "${WARMUP_UPDATES}"
  --lr "${LEARNING_RATE}"
  --weight_decay 0.0
  --token_factor 20
  --slot_tg_depth 1
  --fusion_mode linear_sqrt
  --fusion_reference_count 9
  --loss_version DiceBCE
  --lambda_lpips 0
  --lambda_seg 1
  --lambda_bg_seg 0.25
  --seg_supervision all
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

{
  echo "experiment=all-head-only linear probe from ROI20 checkpoint-802"
  echo "frozen=ViT, collectors, TGEnc, AHER, reconstruction decoder"
  echo "trainable=all nine PatchBinaryHeads and calibration scale/bias"
  echo "gpu_ids=${GPU_IDS} effective_batch=${EFFECTIVE_BATCH}"
  echo "max_updates=${MAX_UPDATES} warmup_updates=${WARMUP_UPDATES} lr=${LEARNING_RATE}"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} | tee "${OUTPUT_DIR}/launcher.log"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=1
"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
