#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${STAGE2_RUN:?}" "${NEW_DATA_ROOT:?}" "${OUTPUT_ROOT:?}" "${STAGE1_TEST:?}"
cd "$EXPERIMENT_WORKDIR"
if [[ -e .git ]]; then
    test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
else
    test "$(<SOURCE_COMMIT)" = "$EXPECTED_COMMIT"
fi
export ARM_F_TARGET=${ARM_F_TARGET:-sifan-xec} ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
CHECKPOINT=$("$PYTHON" tools/incremental44_stage2_checkpoint.py "$STAGE2_RUN")
DATA=$NEW_DATA_ROOT/WORD
COMMON=(--checkpoint "$CHECKPOINT" --class-map configs/orgslot/common8_offline.json
 --class-ids 1,2,3,4,5,6,7,8 --fusion-mode linear_sqrt --batch-size 4 --workers 4
 --min-size 20 --opening-radius 1 --device cuda)
"$PYTHON" -m tools.eval_common8_orgslot_heads "${COMMON[@]}" \
 --data-csv "$DATA/csv/WORD_Validation_2D_native07072.csv" --output-dir "$OUTPUT_ROOT/validation" \
 --split validation --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
"$PYTHON" -m tools.eval_common8_orgslot_heads "${COMMON[@]}" \
 --data-csv "$DATA/csv/WORD_Test_2D_native07072.csv" --output-dir "$OUTPUT_ROOT/test" \
 --split test --binary-threshold 0.5 --class-thresholds "$OUTPUT_ROOT/validation/selected_thresholds.json"
"$PYTHON" -m tools.report_incremental44 --stage1 "$STAGE1_TEST/results.json" \
 --stage2 "$OUTPUT_ROOT/test/results.json" > "$OUTPUT_ROOT/stage_comparison.json"
cat "$OUTPUT_ROOT/stage_comparison.json"
if [[ "${EVALUATE_RECONSTRUCTION:-0}" == 1 ]]; then
 "$PYTHON" -m tools.eval_common8_orgslot_reconstruction_threshold \
  --method orgslot --checkpoint "$CHECKPOINT" --test-csv "$DATA/csv/WORD_Test_2D_native07072.csv" \
  --preprocess-summary "$DATA/metadata/preprocess_summary.json" --expected-spacing 0.7 0.7 2 \
  --output-dir "$OUTPUT_ROOT/reconstruction" --class-map configs/orgslot/common8_offline.json \
  --fusion-mode linear_sqrt --input-size 448 --global-crop-size 448 --batch-size 4 --workers 4 \
  --threshold 0.02 --min-size 20 --opening-radius 1 --class-ids 1,2,3,4,5,6,7,8 --device cuda
 "$PYTHON" tools/validate_word_orgslot_resolution_eval.py --eval-dir "$OUTPUT_ROOT/reconstruction" \
  --expected-input-size 448 --expected-global-crop 448 --expected-spacing 0.7 0.7 2 --expected-cases 24
fi
