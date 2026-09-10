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
MANIFEST_TAG=${MANIFEST_TAG:-native112}
TRAIN_CSV=${TRAIN_CSV:-${PROCESSED_ROOT}/csv/WORD_Training_2D_${MANIFEST_TAG}.csv}
VAL_CSV=${VAL_CSV:-${PROCESSED_ROOT}/csv/WORD_Test_2D_${MANIFEST_TAG}.csv}
PREPROCESS_SUMMARY=${PREPROCESS_SUMMARY:-${PROCESSED_ROOT}/metadata/preprocess_summary.json}
ROI_INDEX=${ROI_INDEX:-${PROCESSED_ROOT}/metadata/small_organ_roi_index.json}
LPIPS_STATE=${LPIPS_STATE:-${DATA_ROOT}/pretrained/owt_lpips_vgg16.pth}
EXPECTED_SPACING=${EXPECTED_SPACING:-"1 1 2"}

FUSION_MODE=${FUSION_MODE:?Set FUSION_MODE after the current AutoPET evaluations}
FUSION_REFERENCE_COUNT=${FUSION_REFERENCE_COUNT:-9}
LAMBDA_SEG=${LAMBDA_SEG:?Set LAMBDA_SEG after the current AutoPET evaluations}
LAMBDA_BG_SEG=${LAMBDA_BG_SEG:-0.25}
SEG_SUPERVISION=${SEG_SUPERVISION:-retained}
SEG_LOSS_TYPE=${SEG_LOSS_TYPE:-dice_bce}
FOCAL_ALPHA=${FOCAL_ALPHA:-0.75}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
TVERSKY_ALPHA_FP=${TVERSKY_ALPHA_FP:-0.3}
TVERSKY_BETA_FN=${TVERSKY_BETA_FN:-0.7}
TVERSKY_EPS=${TVERSKY_EPS:-1e-6}
BALANCED_FOCAL_WEIGHT=${BALANCED_FOCAL_WEIGHT:-0.5}
HARD_NEGATIVE_RATIO=${HARD_NEGATIVE_RATIO:-0.02}
NEGATIVE_SLICE_WEIGHT=${NEGATIVE_SLICE_WEIGHT:-0.1}
AMP_DTYPE=${AMP_DTYPE:-fp16}
CLIP_GRAD=${CLIP_GRAD:-1.0}
FINITE_CHECK_INTERVAL=${FINITE_CHECK_INTERVAL:-50}
SLOT_HEAD_TYPE=${SLOT_HEAD_TYPE:-linear}
SLOT_HEAD_CHANNELS=${SLOT_HEAD_CHANNELS:-128}
DIMENSION=${DIMENSION:-2D}
FIX_FRAME=${FIX_FRAME:-4}
TEMP_STRIDE=${TEMP_STRIDE:-1}
ORGAN_ROI_PROBABILITY=${ORGAN_ROI_PROBABILITY:-0.2}
TGR_MODE=${TGR_MODE:-legacy_batch}
DISABLE_TRAIN_AUGMENTATION=${DISABLE_TRAIN_AUGMENTATION:-0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-}
GPU_IDS=${GPU_IDS:-}
POSITIVE_ROI_LOSS_WEIGHT=${POSITIVE_ROI_LOSS_WEIGHT:-0}
ROI_POSITIVE_SAMPLE_COUNTS=${ROI_POSITIVE_SAMPLE_COUNTS:-}
ROI_FREQUENCY_DATASET_SIZE=${ROI_FREQUENCY_DATASET_SIZE:-}
ROI_FREQUENCY_ALPHA=${ROI_FREQUENCY_ALPHA:-0.5}
ROI_MAX_WEIGHT_RATIO=${ROI_MAX_WEIGHT_RATIO:-4.0}
N_GPU=${N_GPU:-2}
MASTER_PORT=${MASTER_PORT:-25741}
MICRO_BATCH=${MICRO_BATCH:-8}
ACCUM_ITER=${ACCUM_ITER:-12}
TARGET_EFFECTIVE_BATCH=${TARGET_EFFECTIVE_BATCH:-192}
MAX_UPDATES=${MAX_UPDATES:-118800}
WARMUP_UPDATES=${WARMUP_UPDATES:-5940}
BASE_LR=${BASE_LR:-1e-4}
ACTUAL_LR=${ACTUAL_LR:-}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
WORKERS=${WORKERS:-10}
SAVE_FREQ=${SAVE_FREQ:-100}
MAX_STEPS_PER_EPOCH=${MAX_STEPS_PER_EPOCH:-}
NO_SAVE=${NO_SAVE:-0}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
MAE_INIT_CHECKPOINT=${MAE_INIT_CHECKPOINT:-}
MAE_INIT_SCOPE=${MAE_INIT_SCOPE:-encoder_decoder}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Slurm job ${SLURM_JOB_ID} did not provide CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  GPU_BINDING_SOURCE=slurm
elif [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  GPU_BINDING_SOURCE=manual_gpu_ids
else
  GPU_BINDING_SOURCE=inherited_environment
fi

EFFECTIVE_BATCH=$((MICRO_BATCH * ACCUM_ITER * N_GPU))
if [[ "${EFFECTIVE_BATCH}" -ne "${TARGET_EFFECTIVE_BATCH}" ]]; then
  echo "Effective batch ${EFFECTIVE_BATCH} != required ${TARGET_EFFECTIVE_BATCH}" >&2
  exit 2
fi
if [[ "${DIMENSION}" != "2D" && "${DIMENSION}" != "3D" ]]; then
  echo "DIMENSION must be 2D or 3D" >&2
  exit 2
fi
for required in "${PYTHON}" "${TRAIN_CSV}" "${VAL_CSV}" "${PREPROCESS_SUMMARY}" "${LPIPS_STATE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}" >&2
  exit 2
fi
if [[ -n "${MAE_INIT_CHECKPOINT}" && ! -f "${MAE_INIT_CHECKPOINT}" ]]; then
  echo "Missing MAE initialization checkpoint: ${MAE_INIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ -n "${RESUME_CHECKPOINT}" && -n "${MAE_INIT_CHECKPOINT}" ]]; then
  echo "RESUME_CHECKPOINT and MAE_INIT_CHECKPOINT are mutually exclusive" >&2
  exit 2
fi
if [[ "${MAE_INIT_SCOPE}" != "encoder" && "${MAE_INIT_SCOPE}" != "encoder_decoder" ]]; then
  echo "MAE_INIT_SCOPE must be encoder or encoder_decoder" >&2
  exit 2
fi
read -r -a SPACING_ARRAY <<< "${EXPECTED_SPACING}"
if [[ "${#SPACING_ARRAY[@]}" -ne 3 ]]; then
  echo "EXPECTED_SPACING must contain exactly three values" >&2
  exit 2
fi
if [[ "${ARM}" == "roi20" && ! -f "${ROI_INDEX}" ]]; then
  echo "Missing ROI index: ${ROI_INDEX}" >&2
  exit 2
fi

RUN_NAME=${RUN_NAME:-OrgSlot_WORD112_input448_${ARM}_${FUSION_MODE}_seg${LAMBDA_SEG}_eb${TARGET_EFFECTIVE_BATCH}_u${MAX_UPDATES}}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/Results/OrganSlotBank/Common8/WORD_${DIMENSION}/${RUN_NAME}}
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

COMMAND=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${N_GPU}"
  --rdzv_backend=c10d
  --rdzv_endpoint="localhost:${MASTER_PORT}"
  --rdzv_id="${SLURM_JOB_ID:-local}"
  main_pretrain_orgslot_common8_a100.py
  --batch_size "${MICRO_BATCH}"
  --accum_iter "${ACCUM_ITER}"
  --input_size 448
  --dimension "${DIMENSION}"
  --fix_frame "${FIX_FRAME}"
  --temp_stride "${TEMP_STRIDE}"
  --global_crop_size 448
  --roi_crop_size 384
  --expected_spacing "${SPACING_ARRAY[@]}"
  --roi_center_jitter 0.1
  --focus_class_ids 4,5,6
  --max_optimizer_updates "${MAX_UPDATES}"
  --warmup_updates "${WARMUP_UPDATES}"
  --blr "${BASE_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --token_factor 20
  --slot_tg_depth 1
  --slot_head_type "${SLOT_HEAD_TYPE}"
  --slot_head_channels "${SLOT_HEAD_CHANNELS}"
  --pixel_pe "${PIXEL_PE:-none}"
  --query_refinement "${QUERY_REFINEMENT:-none}"
  --segmentation_unit "${SEGMENTATION_UNIT:-slab}"
  --fusion_mode "${FUSION_MODE}"
  --fusion_reference_count "${FUSION_REFERENCE_COUNT}"
  --loss_version L2-LPIPS
  --lambda_lpips 1.0
  --lpips_state "${LPIPS_STATE}"
  --lambda_seg "${LAMBDA_SEG}"
  --lambda_bg_seg "${LAMBDA_BG_SEG}"
  --seg_supervision "${SEG_SUPERVISION}"
  --seg_loss_type "${SEG_LOSS_TYPE}"
  --focal_alpha "${FOCAL_ALPHA}"
  --focal_gamma "${FOCAL_GAMMA}"
  --tversky_alpha_fp "${TVERSKY_ALPHA_FP}"
  --tversky_beta_fn "${TVERSKY_BETA_FN}"
  --tversky_eps "${TVERSKY_EPS}"
  --balanced_focal_weight "${BALANCED_FOCAL_WEIGHT}"
  --hard_negative_ratio "${HARD_NEGATIVE_RATIO}"
  --negative_slice_weight "${NEGATIVE_SLICE_WEIGHT}"
  --amp_dtype "${AMP_DTYPE}"
  --clip_grad "${CLIP_GRAD}"
  --finite_check_interval "${FINITE_CHECK_INTERVAL}"
  --tgr_mode "${TGR_MODE}"
  --positive_roi_loss_weight "${POSITIVE_ROI_LOSS_WEIGHT}"
  --roi_frequency_alpha "${ROI_FREQUENCY_ALPHA}"
  --roi_max_weight_ratio "${ROI_MAX_WEIGHT_RATIO}"
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
if [[ "${POSITIVE_ROI_LOSS_WEIGHT}" != "0" && "${POSITIVE_ROI_LOSS_WEIGHT}" != "0.0" ]]; then
  if [[ -z "${ROI_POSITIVE_SAMPLE_COUNTS}" || -z "${ROI_FREQUENCY_DATASET_SIZE}" ]]; then
    echo "Loss-v3 requires ROI_POSITIVE_SAMPLE_COUNTS and ROI_FREQUENCY_DATASET_SIZE" >&2
    exit 2
  fi
  read -r -a ROI_COUNTS_ARRAY <<< "${ROI_POSITIVE_SAMPLE_COUNTS}"
  COMMAND+=(
    --roi_positive_sample_counts "${ROI_COUNTS_ARRAY[@]}"
    --roi_frequency_dataset_size "${ROI_FREQUENCY_DATASET_SIZE}"
  )
fi
if [[ "${ARM}" == "roi20" ]]; then
  COMMAND+=(
    --organ_roi_aug
    --organ_roi_probability "${ORGAN_ROI_PROBABILITY}"
    --roi_index "${ROI_INDEX}"
  )
fi
if [[ "${DISABLE_TRAIN_AUGMENTATION}" == "1" ]]; then
  COMMAND+=(--disable_train_augmentation)
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  COMMAND+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  COMMAND+=(--max_val_samples "${MAX_VAL_SAMPLES}")
fi
if [[ -n "${MAX_STEPS_PER_EPOCH}" ]]; then
  COMMAND+=(--max_steps_per_epoch "${MAX_STEPS_PER_EPOCH}")
fi
if [[ "${NO_DDP_BUFFER_BROADCAST:-0}" == 1 ]]; then
  COMMAND+=(--no_ddp_buffer_broadcast)
fi
if [[ "${NO_SAVE}" == "1" ]]; then
  COMMAND+=(--no_save)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  COMMAND+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ -n "${ACTUAL_LR}" ]]; then
  COMMAND+=(--lr "${ACTUAL_LR}")
fi
if [[ -n "${MAE_INIT_CHECKPOINT}" ]]; then
  COMMAND+=(
    --mae_init_checkpoint "${MAE_INIT_CHECKPOINT}"
    --mae_init_scope "${MAE_INIT_SCOPE}"
  )
fi

{
  echo "arm=${ARM}"
  echo "data_root=${DATA_ROOT}"
  echo "train_csv=${TRAIN_CSV}"
  echo "gpu_binding_source=${GPU_BINDING_SOURCE}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "spacing=${EXPECTED_SPACING}"
  echo "fusion_mode=${FUSION_MODE} fusion_reference_count=${FUSION_REFERENCE_COUNT} lambda_seg=${LAMBDA_SEG}"
  echo "seg_supervision=${SEG_SUPERVISION} lambda_bg_seg=${LAMBDA_BG_SEG}"
  echo "organ_roi_probability=${ORGAN_ROI_PROBABILITY} tgr_mode=${TGR_MODE}"
  echo "seg_loss_type=${SEG_LOSS_TYPE} focal_alpha=${FOCAL_ALPHA} focal_gamma=${FOCAL_GAMMA}"
  echo "tversky_alpha_fp=${TVERSKY_ALPHA_FP} tversky_beta_fn=${TVERSKY_BETA_FN} tversky_eps=${TVERSKY_EPS}"
  echo "balanced_focal_weight=${BALANCED_FOCAL_WEIGHT} hard_negative_ratio=${HARD_NEGATIVE_RATIO} negative_slice_weight=${NEGATIVE_SLICE_WEIGHT}"
  echo "slot_head_type=${SLOT_HEAD_TYPE} slot_head_channels=${SLOT_HEAD_CHANNELS}"
  echo "amp_dtype=${AMP_DTYPE} clip_grad=${CLIP_GRAD} finite_check_interval=${FINITE_CHECK_INTERVAL}"
  echo "dimension=${DIMENSION} fix_frame=${FIX_FRAME} temp_stride=${TEMP_STRIDE}"
  echo "positive_roi_loss_weight=${POSITIVE_ROI_LOSS_WEIGHT} frequency_alpha=${ROI_FREQUENCY_ALPHA} max_weight_ratio=${ROI_MAX_WEIGHT_RATIO}"
  echo "micro_batch=${MICRO_BATCH} accum_iter=${ACCUM_ITER} effective_batch=${EFFECTIVE_BATCH}"
  echo "base_lr=${BASE_LR} weight_decay=${WEIGHT_DECAY}"
  echo "actual_lr_override=${ACTUAL_LR:-<auto>}"
  echo "max_updates=${MAX_UPDATES} warmup_updates=${WARMUP_UPDATES}"
  echo "resume_checkpoint=${RESUME_CHECKPOINT:-<none>}"
  echo "mae_init_checkpoint=${MAE_INIT_CHECKPOINT:-<none>} mae_init_scope=${MAE_INIT_SCOPE}"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} | tee "${OUTPUT_DIR}/launcher.log"

cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1
"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
