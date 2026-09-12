#!/usr/bin/env bash
# Explicit account paths. Sourced by the frozen training and evaluation scripts.
set -euo pipefail
: "${ARM_F_TARGET:?}" "${ARM_F_DIMENSION:?}" "${EXPERIMENT_WORKDIR:?}"
case "$ARM_F_TARGET" in
  bolin-sip)
    export PYTHON=/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python
    export PROCESSED_ROOT=/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8_07072_NATIVE/WORD
    export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index.json
    export LPIPS_STATE=/gpfs/work/aac/bolinren19/OD_OWT_orgslot/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
    ;;
  sifan-xec)
    export PYTHON=/gpfs/work/aac/sifansong/envs/abdpet/bin/python
    export PROCESSED_ROOT=/gpfs/work/aac/sifansong/data/OWT_Common8_07072_NATIVE/WORD
    export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_verified.json
    export LPIPS_STATE=/gpfs/work/aac/sifansong/data/pretrained/owt_lpips_vgg16.pth
    ;;
  anteng-xec)
    export PYTHON=/gpfs/work/aac/antengcai23/envs/abdpet/bin/python
    export PROCESSED_ROOT=/gpfs/work/aac/antengcai23/data/OWT_Common8_07072_NATIVE/WORD
    export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_verified.json
    export LPIPS_STATE=/gpfs/work/aac/antengcai23/worktrees/orgslotbank/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
    ;;
  *) exit 2 ;;
esac
export DIMENSION=$ARM_F_DIMENSION MANIFEST_TAG=native07072 EXPECTED_SPACING="0.7 0.7 2"
export PATH="$(dirname "$PYTHON"):$PATH" PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}"
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=small_organ FOCAL_ALPHA=0.75 FOCAL_GAMMA=2
export TVERSKY_ALPHA_FP=0.3 TVERSKY_BETA_FN=0.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=0.5 HARD_NEGATIVE_RATIO=0.02 NEGATIVE_SLICE_WEIGHT=0.1
export SLOT_HEAD_TYPE=arm_f_attention SLOT_HEAD_CHANNELS=128 QUERY_REFINEMENT=none PIXEL_PE=none
export SEGMENTATION_UNIT=slice ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export AMP_DTYPE=bf16 CLIP_GRAD=1 FINITE_CHECK_INTERVAL=25
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 ACTUAL_LR=7.5e-5 WEIGHT_DECAY=0.05
export N_GPU=4 WORKERS=4 MASTER_PORT=0 POSITIVE_ROI_LOSS_WEIGHT=0
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-} MAE_INIT_CHECKPOINT=""
export BACKGROUND_REDUCTION=${BACKGROUND_REDUCTION:-mean}
case "$DIMENSION" in
  2D) export MICRO_BATCH=8 ACCUM_ITER=6 TARGET_EFFECTIVE_BATCH=192 SAVE_FREQ=10 ;;
  3D) export MICRO_BATCH=4 ACCUM_ITER=3 TARGET_EFFECTIVE_BATCH=48 SAVE_FREQ=10 FIX_FRAME=4 TEMP_STRIDE=1 ;;
  *) exit 2 ;;
esac
