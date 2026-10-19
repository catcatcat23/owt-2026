#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPERIMENT_COMMIT:?}" "${BACKBONE:?}"
: "${CNN_INIT_CHECKPOINT:?}" "${HSAM_SUPERVISION:?}" "${RUN_NAME:?}"
case "$HSAM_SUPERVISION" in soft_prior_aux|downsample_gt) ;; *) exit 2 ;; esac
cd "$EXPERIMENT_WORKDIR"
if [[ -e .git ]]; then
 test "$(git rev-parse HEAD)" = "$EXPERIMENT_COMMIT"
else
 test "$(<SOURCE_COMMIT)" = "$EXPERIMENT_COMMIT"
fi
export OUTPUT_DIR="$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME"
test ! -e "$OUTPUT_DIR"
export PYTHON=${CNN_PYTHON:-/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python}
export PATH="$(dirname "$PYTHON"):$PATH" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
export PROCESSED_ROOT=${CNN_WORD_ROOT:-/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8_07072_NATIVE/WORD}
export MANIFEST_TAG=native07072 EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX=${CNN_ROI_INDEX:-$PROCESSED_ROOT/metadata/small_organ_roi_index.json}
export LPIPS_STATE=${CNN_LPIPS_STATE:-/gpfs/work/aac/bolinren19/OD_OWT_orgslot/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth}
export SLOT_HEAD_TYPE=arm_f_sam_tail SLOT_HEAD_CHANNELS=128 QUERY_REFINEMENT=none PIXEL_PE=none
export DIMENSION=2D FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export CNN_SKIP_SOURCE=encoder
export LAMBDA_SEG=.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained SEG_LOSS_TYPE=small_organ
export FOCAL_ALPHA=.75 FOCAL_GAMMA=2 TVERSKY_ALPHA_FP=.3 TVERSKY_BETA_FN=.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=.5 HARD_NEGATIVE_RATIO=.02 NEGATIVE_SLICE_WEIGHT=.1
export BACKGROUND_REDUCTION=topk ORGAN_ROI_PROBABILITY=.2 TGR_MODE=legacy_batch
export AMP_DTYPE=bf16 CLIP_GRAD=1 FINITE_CHECK_INTERVAL=25 SEGMENTATION_UNIT=slice
export N_GPU=4 MICRO_BATCH=16 ACCUM_ITER=3 TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 ACTUAL_LR=7.5e-5 WEIGHT_DECAY=.05
export WORKERS=4 SAVE_FREQ=100 MAE_INIT_CHECKPOINT= RESUME_CHECKPOINT= INIT_CHECKPOINT=
if [[ -n "${CNN_RESUME_SOURCE:-}" ]]; then
 export RESUME_CHECKPOINT=$("$PYTHON" tools/select_cnn_resume.py "$CNN_RESUME_SOURCE" "$BACKBONE" "$HSAM_SUPERVISION")
 export CNN_INIT_CHECKPOINT= SAVE_FREQ=10
 if [[ "$RESUME_CHECKPOINT" == */checkpoint-802.pth ]]; then
  mkdir -p "$OUTPUT_DIR"
  cp "$RESUME_CHECKPOINT" "$OUTPUT_DIR/checkpoint-802.pth"
  cp "$CNN_RESUME_SOURCE/cnn_initial_checkpoint_load.json" "$OUTPUT_DIR/"
  echo 'Source already completed; forwarded final checkpoint without retraining'
  exit 0
 fi
fi
export MASTER_PORT=0
echo "commit=$EXPERIMENT_COMMIT runtime=$EXPERIMENT_WORKDIR init=$CNN_INIT_CHECKPOINT"
if [[ -z "${CNN_RESUME_SOURCE:-}" ]]; then test -s "$CNN_INIT_CHECKPOINT"; fi
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT" --roi_index "$ROI_INDEX"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -s "$OUTPUT_DIR/checkpoint-802.pth"
if [[ -n "${CNN_RESUME_SOURCE:-}" ]]; then
 cp "$CNN_RESUME_SOURCE/cnn_initial_checkpoint_load.json" "$OUTPUT_DIR/"
fi
test -s "$OUTPUT_DIR/cnn_initial_checkpoint_load.json"
