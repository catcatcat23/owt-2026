#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
test "$(cat SOURCE_COMMIT)" = "${EXPECTED_COMMIT:?}"
/usr/bin/python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["body_clipped_cases"] == [], "BTCV body crop coverage audit failed"' "$RUN_ROOT/body_center_audit.json"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=2D ARM_F_HEAD_TYPE=arm_f_sam_tail
export ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=6 BACKGROUND_REDUCTION=topk
source scripts/orgslot/arm_f_environment.sh
export PROCESSED_ROOT=${BTCV_ROOT:?} MANIFEST_TAG=native07072 EXPECTED_SPACING='0.7 0.7 2'
export TRAIN_CSV=$BTCV_ROOT/csv/BTCV_Training_2D_native07072.csv
# No held-out validation exists in the 24/6 protocol. Never use test labels in training.
export VAL_CSV=$TRAIN_CSV NO_VALIDATION=1
export ROI_INDEX=$BTCV_ROOT/metadata/small_organ_roi_index.json
export PREPROCESS_SUMMARY=$BTCV_ROOT/metadata/preprocess_summary.json
export HSAM_SUPERVISION=soft_prior_aux MAE_INIT_SCOPE=encoder
export INPUT_SIZE=512 GLOBAL_CROP_SIZE=512 BODY_CENTER=1
export N_GPU=2 ACCUM_ITER=6 MICRO_BATCH=16 SAVE_FREQ=500
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT"
checkpoint=$("$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --resume)
if [[ -n "$checkpoint" ]]; then
 if "$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --final > "$RUN_ROOT/final_checkpoint.txt"; then exit 0; fi
 export RESUME_CHECKPOINT=$checkpoint MAE_INIT_CHECKPOINT=''
else
 [[ "${1:-initial}" == initial ]] || { echo 'Continuation has no checkpoint; refuse scratch restart'; exit 2; }
 export MAE_INIT_CHECKPOINT=${MAE_CHECKPOINT:?} RESUME_CHECKPOINT=''
 printf '0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  %s\n' "$MAE_INIT_CHECKPOINT" | sha256sum -c -
fi
export RUN_NAME=BTCV512_SAM_softaux_${SLURM_JOB_ID} OUTPUT_DIR=$RUN_ROOT/train_${SLURM_JOB_ID}
bash scripts/orgslot/run_word_112_448_a100.sh roi20
"$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --final > "$RUN_ROOT/final_checkpoint.txt"
