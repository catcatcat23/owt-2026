#!/usr/bin/env bash
set -euo pipefail
stage=${1:?stage 1..4}
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${RUN_ROOT:?}" "${WORD_ROOT:?}" "${MAE_CHECKPOINT:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(cat SOURCE_COMMIT)" = "$EXPECTED_COMMIT"
export ARM_F_TARGET=anteng-xec ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
# No GPU smoke test. Actual training starts directly after input validation.
"$PYTHON" tools/prepare_word_official96.py validate --plan configs/orgslot/word_official96_seed42.json --output "$(dirname "$WORD_ROOT")"
output="$RUN_ROOT/stage$stage"
mkdir -p "$output"
if [[ -s "$output/final_checkpoint.txt" ]]; then
 "$PYTHON" -c 'import sys; from util.orgslot_incremental2222 import final_checkpoint; print(final_checkpoint(sys.argv[1], int(sys.argv[2])))' "$output" "$stage"
 exit 0
fi
args=(--stage_index "$stage" --output_dir "$output" --save_freq 10)
if [[ "$stage" == 1 ]]; then
 printf '0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  %s\n' "$MAE_CHECKPOINT" | sha256sum -c -
 args+=(--mae_init_checkpoint "$MAE_CHECKPOINT" --lpips_state "$LPIPS_STATE")
else
 previous="$RUN_ROOT/stage$((stage-1))"
 checkpoint=$("$PYTHON" -c 'import sys; from util.orgslot_incremental2222 import final_checkpoint; print(final_checkpoint(sys.argv[1], int(sys.argv[2])))' "$previous" "$((stage-1))")
 args+=(--previous_checkpoint "$checkpoint")
fi
# Checkpoint selection is by epoch/budget, never test score. Corruption stops the chain.
resume=$("$PYTHON" tools/incremental2222_checkpoint.py "$output" "$stage" --resume)
if [[ -n "$resume" ]]; then args+=(--resume "$resume"); fi
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
 main_pretrain_orgslot_incremental2222.py "${args[@]}" --batch_size 16 --accum_iter 3 \
 --data_path "$WORD_ROOT/csv/WORD_Training_2D_native07072.csv" \
 --val_data_path "$WORD_ROOT/csv/WORD_Validation_2D_native07072.csv" \
 --test_data_path "$WORD_ROOT/csv/WORD_Test_2D_native07072.csv" \
 --preprocess_summary "$WORD_ROOT/metadata/preprocess_summary.json" \
 --roi_index "$WORD_ROOT/metadata/small_organ_roi_index.json"

"$PYTHON" tools/incremental2222_checkpoint.py "$output" "$stage" > "$output/final_checkpoint.txt.tmp"
mv "$output/final_checkpoint.txt.tmp" "$output/final_checkpoint.txt"
