#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
test "$(cat SOURCE_COMMIT)" = "${EXPECTED_COMMIT:?}"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=2D ARM_F_HEAD_TYPE=arm_f_sam_tail
source scripts/orgslot/arm_f_environment.sh
export BACKGROUND_REDUCTION=topk HSAM_SUPERVISION=soft_prior_aux MAE_INIT_SCOPE=encoder
export PROCESSED_ROOT=${BTCV_ROOT:?} MANIFEST_TAG=native07072 EXPECTED_SPACING='0.7 0.7 2'
export TRAIN_CSV=$BTCV_ROOT/csv/BTCV_Training_2D_native07072.csv VAL_CSV=$BTCV_ROOT/csv/BTCV_Training_2D_native07072.csv NO_VALIDATION=1
export ROI_INDEX=$BTCV_ROOT/metadata/all_organ_roi_index.json PREPROCESS_SUMMARY=$BTCV_ROOT/metadata/preprocess_summary.json
export INPUT_SIZE=512 GLOBAL_CROP_SIZE=512 BODY_CENTER=1 N_GPU=4 MICRO_BATCH=16 ACCUM_ITER=3 SAVE_FREQ=100
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export ROI_APPEND_FRACTION=0 ROI_FIT_TARGET=0 FOCUS_CLASS_IDS=4,5,6 BACKBONE=vit CNN_INIT_CHECKPOINT= MAE_INIT_CHECKPOINT=$MAE_CHECKPOINT
case "${VARIANT:?}" in
 vit_current) export ROI_INDEX=$BTCV_ROOT/metadata/small_organ_roi_index.json ;;
 vit_append) export ROI_APPEND_FRACTION=0.25 ROI_FIT_TARGET=1 FOCUS_CLASS_IDS=1,2,3,4,5,6,7,8 ;;
 resenc_append) export ROI_APPEND_FRACTION=0.25 ROI_FIT_TARGET=1 FOCUS_CLASS_IDS=1,2,3,4,5,6,7,8 BACKBONE=nnunet_resenc_s16 CNN_SKIP_SOURCE=encoder CNN_INIT_CHECKPOINT=$RESENC_CHECKPOINT MAE_INIT_CHECKPOINT= ;;
 *) exit 2 ;;
esac
mkdir -p "$RUN_ROOT"
checkpoint=$("$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --resume)
if [[ -n "$checkpoint" ]]; then
 if "$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --final > "$RUN_ROOT/final_checkpoint.txt"; then exit 0; fi
 export RESUME_CHECKPOINT=$checkpoint MAE_INIT_CHECKPOINT= CNN_INIT_CHECKPOINT=
else
 [[ "${1:-initial}" == initial ]] || { echo 'No checkpoint: refusing scratch continuation'; exit 2; }
 export RESUME_CHECKPOINT=
fi
export OUTPUT_DIR=$RUN_ROOT/train_${SLURM_JOB_ID} RUN_NAME=BTCV_seed43_${VARIANT}_${SLURM_JOB_ID}
bash scripts/orgslot/run_word_112_448_a100.sh roi20
"$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --final > "$RUN_ROOT/final_checkpoint.txt"
