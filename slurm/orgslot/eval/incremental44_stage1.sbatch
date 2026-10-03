#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${STAGE1_RUN:?}" "${NEW_DATA_ROOT:?}" "${OUTPUT_ROOT:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
export ARM_F_TARGET=bolin-sip ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
CHECKPOINT=$("$PYTHON" tools/incremental44_final_checkpoint.py "$STAGE1_RUN")
DATA=$NEW_DATA_ROOT/WORD
COMMON=(--checkpoint "$CHECKPOINT" --class-map configs/orgslot/common8_incremental44.json
 --stage1 --class-ids 1,2,3,4 --fusion-mode linear_sqrt --batch-size 4 --workers 4
 --min-size 20 --opening-radius 1 --device cuda)
"$PYTHON" -m tools.eval_common8_orgslot_heads "${COMMON[@]}" \
 --data-csv "$DATA/csv/WORD_Validation_2D_native07072.csv" --output-dir "$OUTPUT_ROOT/validation" \
 --split validation --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
"$PYTHON" -m tools.eval_common8_orgslot_heads "${COMMON[@]}" \
 --data-csv "$DATA/csv/WORD_Test_2D_native07072.csv" --output-dir "$OUTPUT_ROOT/test" \
 --split test --binary-threshold 0.5 --class-thresholds "$OUTPUT_ROOT/validation/selected_thresholds.json"
"$PYTHON" - "$OUTPUT_ROOT/test/results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d['complete_split'] and d['checkpoint_load']['exact']
for mode in ('binary_post', 'selected_post'):
    assert all(d['metrics'][mode][str(c)]['case_count'] == 24 for c in range(5))
    print(mode, 'old-four mean Dice', d['metrics'][mode]['foreground_mean_case_dice_presence_mean'])
print('Stage1 old-four and derived-background evaluation complete')
PY
