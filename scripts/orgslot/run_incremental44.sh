#!/usr/bin/env bash
set -euo pipefail
# Run from the pinned repository root, inside an allocated GPU job.
# No sbatch or implicit cross-account paths here.
stage=${1:?Usage: bash scripts/orgslot/run_incremental44.sh stage1|stage2 [extra arguments]}
shift
case "$stage" in stage1|stage2) ;; *) echo 'Expected stage1 or stage2' >&2; exit 2 ;; esac
: "${WORD_ROOT:?Set this account own WORD data root}"
: "${OUTPUT_DIR:?Use a fresh output directory per stage/background variant}"
: "${PYTHON_BIN:?Set this account own Python executable}"
extra=()
if [[ "$stage" == stage1 ]]; then
    : "${MAE_CHECKPOINT:?Set the AutoPET MAE checkpoint, not a WORD checkpoint}"
    : "${LPIPS_STATE:?Set this account LPIPS state file}"
    extra+=(--mae_init_checkpoint "$MAE_CHECKPOINT" --lpips_state "$LPIPS_STATE")
else
    : "${STAGE1_CHECKPOINT:?Set the completed incremental stage1 checkpoint}"
    extra+=(--stage1_checkpoint "$STAGE1_CHECKPOINT" --background_policy "${BACKGROUND_POLICY:-separation}")
    extra+=(--stage2_shared_segmentation "${STAGE2_SHARED_SEGMENTATION:-frozen}")
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    main_pretrain_orgslot_incremental44.py \
    --incremental_stage "$stage" \
    --data_path "$WORD_ROOT/csv/WORD_Training_2D_native07072.csv" \
    --val_data_path "$WORD_ROOT/csv/WORD_Validation_2D_native07072.csv" \
    --test_data_path "$WORD_ROOT/csv/WORD_Test_2D_native07072.csv" \
    --preprocess_summary "$WORD_ROOT/metadata/preprocess_summary.json" \
    --roi_index "$WORD_ROOT/metadata/small_organ_roi_index.json" \
    --output_dir "$OUTPUT_DIR" "${extra[@]}" "$@"
