#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPERIMENT_COMMIT:?}" "${BACKBONE:?}"
: "${CNN_INIT_CHECKPOINT:?}" "${HSAM_SUPERVISION:?}" "${RUN_NAME:?}"
case "$HSAM_SUPERVISION" in soft_prior_aux|downsample_gt) ;; *) exit 2 ;; esac
cd "$EXPERIMENT_WORKDIR"
test "$(git rev-parse HEAD)" = "$EXPERIMENT_COMMIT"
export OUTPUT_DIR="$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME"
test ! -e "$OUTPUT_DIR"
export PYTHON=/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python
export PATH="$(dirname "$PYTHON"):$PATH" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
export PROCESSED_ROOT=/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8_07072_NATIVE/WORD
export MANIFEST_TAG=native07072 EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX="$PROCESSED_ROOT/metadata/small_organ_roi_index.json"
export LPIPS_STATE=/gpfs/work/aac/bolinren19/OD_OWT_orgslot/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export SLOT_HEAD_TYPE=arm_f_sam_tail SLOT_HEAD_CHANNELS=128 QUERY_REFINEMENT=none PIXEL_PE=none
export DIMENSION=2D FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained SEG_LOSS_TYPE=small_organ
export FOCAL_ALPHA=.75 FOCAL_GAMMA=2 TVERSKY_ALPHA_FP=.3 TVERSKY_BETA_FN=.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=.5 HARD_NEGATIVE_RATIO=.02 NEGATIVE_SLICE_WEIGHT=.1
export BACKGROUND_REDUCTION=topk ORGAN_ROI_PROBABILITY=.2 TGR_MODE=legacy_batch
export AMP_DTYPE=bf16 CLIP_GRAD=1 FINITE_CHECK_INTERVAL=25 SEGMENTATION_UNIT=slice
export N_GPU=4 MICRO_BATCH=16 ACCUM_ITER=3 TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 ACTUAL_LR=7.5e-5 WEIGHT_DECAY=.05
export WORKERS=4 SAVE_FREQ=100 MAE_INIT_CHECKPOINT= RESUME_CHECKPOINT= INIT_CHECKPOINT=
export MASTER_PORT=0
echo "commit=$EXPERIMENT_COMMIT runtime=$EXPERIMENT_WORKDIR init=$CNN_INIT_CHECKPOINT"
test -s "$CNN_INIT_CHECKPOINT"
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT" --roi_index "$ROI_INDEX"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -s "$OUTPUT_DIR/checkpoint-802.pth"
test -s "$OUTPUT_DIR/cnn_initial_checkpoint_load.json"
