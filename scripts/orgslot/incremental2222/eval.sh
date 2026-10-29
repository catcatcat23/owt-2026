#!/usr/bin/env bash
set -euo pipefail
stage=${1:?stage}
cd "${EXPERIMENT_WORKDIR:?}"
test "$(cat SOURCE_COMMIT)" = "${EXPECTED_COMMIT:?}"
export ARM_F_TARGET=anteng-xec ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
checkpoint=$("$PYTHON" tools/incremental2222_checkpoint.py "$RUN_ROOT/stage$stage" "$stage")
output="$RUN_ROOT/stage${stage}_evaluation"
mkdir -p "$output"
classmap="$output/classes.json"
"$PYTHON" - "$stage" "$classmap" <<'PY'
import json,sys
from util.orgslot_incremental2222 import seen
classes=json.load(open('configs/orgslot/common8_incremental2222.json'))['classes']
json.dump({'incremental_protocol':'2222','stage_index':int(sys.argv[1]), 'classes':classes[:len(seen(int(sys.argv[1])))], 'stages':{'base':{'stage':'evaluation','visible_slots':list(seen(int(sys.argv[1])))}}},open(sys.argv[2],'w'),indent=2)
PY
ids=$(seq -s, 1 "$((2*stage))")
common=(--checkpoint "$checkpoint" --class-map "$classmap" --class-ids "$ids"
 --fusion-mode linear_sqrt --batch-size 4 --workers 4 --min-size 20 --opening-radius 1 --device cuda)
"$PYTHON" -m tools.eval_common8_orgslot_heads "${common[@]}" \
 --data-csv "$WORD_ROOT/csv/WORD_Validation_2D_native07072.csv" --output-dir "$output/validation" \
 --split validation --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
"$PYTHON" -m tools.eval_common8_orgslot_heads "${common[@]}" \
 --data-csv "$WORD_ROOT/csv/WORD_Test_2D_native07072.csv" --output-dir "$output/test" \
 --split test --binary-threshold 0.5 --class-thresholds "$output/validation/selected_thresholds.json"
"$PYTHON" -m tools.eval_common8_orgslot_reconstruction_threshold \
 --method orgslot --checkpoint "$checkpoint" --test-csv "$WORD_ROOT/csv/WORD_Test_2D_native07072.csv" \
 --preprocess-summary "$WORD_ROOT/metadata/preprocess_summary.json" --expected-spacing 0.7 0.7 2 \
 --output-dir "$output/reconstruction" --class-map "$classmap" --fusion-mode linear_sqrt \
 --input-size 448 --global-crop-size 448 --batch-size 4 --workers 4 --threshold 0.02 \
 --min-size 20 --opening-radius 1 --class-ids "$ids" --device cuda
"$PYTHON" tools/report_incremental2222.py "$output/test/results.json" "$stage" > "$output/group_scores.json"
