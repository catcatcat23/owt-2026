#!/usr/bin/env bash
# Explicit checkpoint evaluation; never launches subsequent training stages.
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
test "$(cat SOURCE_COMMIT)" = "${EXPECTED_COMMIT:?}"
export ARM_F_TARGET=bolin-sip ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${RESUMABLE_EVALUATION:-0}" == 1 ]]; then
  export EVAL_DEADLINE_EPOCH=$(( $(date +%s) + 3000 ))
fi
run_eval() {
  local status=0
  "$@" || status=$?
  if [[ "$status" == 75 && "${RESUMABLE_EVALUATION:-0}" == 1 ]]; then
    exit 0  # Successful checkpoint yield; afterok successor resumes it.
  fi
  return "$status"
}
resume_args=()
if [[ "${RESUMABLE_EVALUATION:-0}" == 1 ]]; then
  resume_args=(--resume-evaluation)
fi
: "${CHECKPOINT:?}" "${CLASS_MAP:?}" "${DATA_ROOT:?}" "${EVAL_OUTPUT:?}"
mkdir -p "$EVAL_OUTPUT"
"$PYTHON" - "$CLASS_MAP" "$EVAL_OUTPUT/classes.json" <<'PYMAP'
import json, sys
config = json.load(open(sys.argv[1]))
assert [c["raw_id"] for c in config["classes"]] == list(range(9))
config["stages"] = {"base": {"stage": "evaluation", "visible_slots": [c["name"] for c in config["classes"]]}}
with open(sys.argv[2], "w") as handle:
    json.dump(config, handle, indent=2)
PYMAP
CLASS_MAP=$EVAL_OUTPUT/classes.json
common=(--checkpoint "$CHECKPOINT" --class-map "$CLASS_MAP" --class-ids 1,2,3,4,5,6,7,8 --fusion-mode linear_sqrt --input-size 448 --global-crop-size 448 --batch-size 4 --workers 4 --min-size 20 --opening-radius 1 --device cuda)
run_eval "$PYTHON" -m tools.eval_common8_orgslot_heads "${resume_args[@]}" "${common[@]}" --data-csv "$DATA_ROOT/csv/WORD_Validation_2D_native07072.csv" --output-dir "$EVAL_OUTPUT/validation" --split validation --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
run_eval "$PYTHON" -m tools.eval_common8_orgslot_heads "${resume_args[@]}" "${common[@]}" --data-csv "$DATA_ROOT/csv/WORD_Test_2D_native07072.csv" --output-dir "$EVAL_OUTPUT/test" --split test --binary-threshold 0.5 --class-thresholds "$EVAL_OUTPUT/validation/selected_thresholds.json"
run_eval "$PYTHON" -m tools.eval_common8_orgslot_reconstruction_threshold "${resume_args[@]}" --method orgslot --checkpoint "$CHECKPOINT" --class-map "$CLASS_MAP" --class-ids 1,2,3,4,5,6,7,8 --input-size 448 --global-crop-size 448 --test-csv "$DATA_ROOT/csv/WORD_Test_2D_native07072.csv" --preprocess-summary "$DATA_ROOT/metadata/preprocess_summary.json" --expected-spacing 0.7 0.7 2 --output-dir "$EVAL_OUTPUT/reconstruction" --fusion-mode linear_sqrt --batch-size 4 --workers 4 --threshold 0.02 --min-size 20 --opening-radius 1 --device cuda
