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
case "${INCREMENTAL_ARCHITECTURE:-e_cross}" in
 e_cross) ;;
 sam_soft_aux)
    extra+=(--slot_head_type arm_f_sam_tail --query_refinement none --hsam_supervision soft_prior_aux)
    if [[ "$stage" == stage2 ]]; then
        export STAGE2_SHARED_SEGMENTATION=${STAGE2_SHARED_SEGMENTATION:-slot_private}
    fi
    ;;
 *) echo 'Invalid INCREMENTAL_ARCHITECTURE' >&2; exit 2 ;;
esac
if [[ "$stage" == stage1 ]]; then
    : "${MAE_CHECKPOINT:?Set the AutoPET MAE checkpoint, not a WORD checkpoint}"
    : "${LPIPS_STATE:?Set this account LPIPS state file}"
    extra+=(--mae_init_checkpoint "$MAE_CHECKPOINT" --lpips_state "$LPIPS_STATE")
else
    : "${STAGE1_CHECKPOINT:?Set the completed incremental stage1 checkpoint}"
    extra+=(--stage1_checkpoint "$STAGE1_CHECKPOINT" --background_policy "${BACKGROUND_POLICY:-separation}")
    extra+=(--stage2_shared_segmentation "${STAGE2_SHARED_SEGMENTATION:-frozen}")
    extra+=(--stage2_readout_adapter "${STAGE2_READOUT_ADAPTER:-none}")
    extra+=(--stage2_pixel_decoder "${STAGE2_PIXEL_DECODER:-shared}")
fi
nproc=${INCREMENTAL_NPROC_PER_NODE:-4}
micro_batch=${ARM_F_MICRO_BATCH:-16}
accum_iter=${ARM_F_ACCUM_ITER:-3}
[[ "$nproc" =~ ^[1-9][0-9]*$ && "$micro_batch" =~ ^[1-9][0-9]*$ && "$accum_iter" =~ ^[1-9][0-9]*$ ]] || exit 2
if (( nproc * micro_batch * accum_iter != 192 )); then
    echo 'Incremental training requires effective batch 192' >&2
    exit 2
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$nproc" \
    main_pretrain_orgslot_incremental44.py \
    --incremental_stage "$stage" \
    --data_path "$WORD_ROOT/csv/WORD_Training_2D_native07072.csv" \
    --val_data_path "$WORD_ROOT/csv/WORD_Validation_2D_native07072.csv" \
    --test_data_path "$WORD_ROOT/csv/WORD_Test_2D_native07072.csv" \
    --preprocess_summary "$WORD_ROOT/metadata/preprocess_summary.json" \
    --roi_index "$WORD_ROOT/metadata/small_organ_roi_index.json" \
    --output_dir "$OUTPUT_DIR" --batch_size "$micro_batch" --accum_iter "$accum_iter" "${extra[@]}" "$@"
