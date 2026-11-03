#!/usr/bin/env bash
set -euo pipefail
stage=${1:?stage}
mode=${2:-initial}
source "${EXPERIMENT_WORKDIR:?}/scripts/orgslot/pcdd_suite/environment.sh"
output=$RUN_ROOT/stage$stage
mkdir -p "$output"
selector=("$PYTHON" tools/incremental2222_checkpoint.py "$output" "$stage" --protocol "$PROTOCOL")
# A complete predecessor allocation is a no-op; validate rather than trusting a marker.
if "${selector[@]}" > "$output/final_checkpoint.txt.tmp" 2> "$output/completion_check.log"; then
 mv "$output/final_checkpoint.txt.tmp" "$output/final_checkpoint.txt"
 exit 0
fi
resume=$("${selector[@]}" --resume)
[[ -n "$resume" || "$mode" == initial ]] || { echo 'No checkpoint for continuation'; exit 2; }
args=(--protocol "$PROTOCOL" --dataset_protocol "$DATASET" --visibility_config "$CONFIG"
 --stage_index "$stage" --output_dir "$output" --save_freq 50 --roi_focus_policy "${ROI_FOCUS_POLICY:-small}"
 --data_path "$TRAIN_CSV" --val_data_path "$VAL_CSV" --test_data_path "$TEST_CSV"
 --roi_index "${SUITE_ROI_INDEX:-$DATA_ROOT/metadata/small_organ_roi_index.json}"
 --preprocess_summary "$DATA_ROOT/metadata/preprocess_summary.json" --batch_size 16 --accum_iter 3)
if [[ "$stage" == 1 ]]; then
 printf '0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  %s\n' "$MAE_CHECKPOINT" | sha256sum -c -
 args+=(--mae_init_checkpoint "$MAE_CHECKPOINT" --lpips_state "$LPIPS_STATE")
else
 previous=$("$PYTHON" tools/incremental2222_checkpoint.py "$RUN_ROOT/stage$((stage-1))" "$((stage-1))" --protocol "$PROTOCOL")
 args+=(--previous_checkpoint "$previous")
fi
[[ -z "$resume" ]] || args+=(--resume "$resume")
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
 main_pretrain_orgslot_incremental2222.py "${args[@]}" "${TRAIN_GEOMETRY[@]}"
"${selector[@]}" > "$output/final_checkpoint.txt.tmp"
mv "$output/final_checkpoint.txt.tmp" "$output/final_checkpoint.txt"
