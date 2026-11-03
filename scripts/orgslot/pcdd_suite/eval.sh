#!/usr/bin/env bash
set -euo pipefail
stage=${1:?stage}
source "${EXPERIMENT_WORKDIR:?}/scripts/orgslot/pcdd_suite/environment.sh"
checkpoint=$("$PYTHON" tools/incremental2222_checkpoint.py "$RUN_ROOT/stage$stage" "$stage" --protocol "$PROTOCOL")
output=$RUN_ROOT/stage${stage}_evaluation
mkdir -p "$output"
classmap=$output/classes.json
ids=$("$PYTHON" - "$PROTOCOL" "$stage" "$CONFIG" "$classmap" <<'PY'
import json,sys
from util.orgslot_incremental2222 import seen
protocol,stage,config,target=sys.argv[1:];names=seen(int(stage),protocol)
classes=json.load(open(config))['classes'][:len(names)]
json.dump({'incremental_protocol':protocol,'stage_index':int(stage),'classes':classes,'stages':{'base':{'stage':'evaluation','visible_slots':list(names)}}},open(target,'w'),indent=2)
print(','.join(str(i) for i in range(1,len(names))))
PY
)
calibration=validation
[[ "$DATASET" != BTCV ]] || calibration=train_calibration
common=(--checkpoint "$checkpoint" --class-map "$classmap" --class-ids "$ids" --fusion-mode linear_sqrt --batch-size 4 --workers 4 --min-size 20 --opening-radius 1 --device cuda)
"$PYTHON" -m tools.eval_common8_orgslot_heads "${common[@]}" "${GEOMETRY[@]}" --data-csv "$VAL_CSV" --output-dir "$output/$calibration" --split "$calibration" --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
"$PYTHON" -m tools.eval_common8_orgslot_heads "${common[@]}" "${GEOMETRY[@]}" --data-csv "$TEST_CSV" --output-dir "$output/test" --split test --binary-threshold 0.5 --class-thresholds "$output/$calibration/selected_thresholds.json"
"$PYTHON" -m tools.eval_common8_orgslot_reconstruction_threshold --method orgslot --checkpoint "$checkpoint" --class-map "$classmap" --class-ids "$ids" "${GEOMETRY[@]}" --test-csv "$TEST_CSV" --preprocess-summary "$DATA_ROOT/metadata/preprocess_summary.json" --expected-spacing 0.7 0.7 2 --output-dir "$output/reconstruction" --fusion-mode linear_sqrt --batch-size 4 --workers 4 --threshold 0.02 --min-size 20 --opening-radius 1 --device cuda
"$PYTHON" tools/report_pcdd_suite.py "$output/test/results.json" --protocol "$PROTOCOL" --stage "$stage" --dataset "$DATASET" --offline "${OFFLINE_RESULT:-$SUITE_ROOT/$DATASET/pcdd_offline/stage1_evaluation/test/results.json}" > "$output/comparison.json"

# Queue the next stage only after all metrics/reports above succeed.
if [[ -n "${SUITE_PLAN:-}" ]]; then
 export SUITE_INDEX=$((SUITE_INDEX + 1)) PREDECESSOR_JOB=${SLURM_JOB_ID:?}
 export HOLD_FIRST=0
 bash scripts/orgslot/pcdd_suite/queue.sh
fi
