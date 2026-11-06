#!/usr/bin/env bash
# New suite root required. Use current-stage foreground organs only, never old GT.
set -euo pipefail
: "${SUITE_ROI_INDEX:?}" "${DATASET:?}"
export ROI_FOCUS_POLICY=current
python3 - "$SUITE_ROI_INDEX" <<'PY'
import json,sys
index=json.load(open(sys.argv[1]))
assert all(index['class_indices'].get(str(i)) for i in range(1,9)), 'Missing organ ROI pool'
PY
if [[ "$DATASET" == WORD ]]; then
 # Stage1 is the completed all-current-organ ROI experiment, linked in advance.
 test -s "$SUITE_ROOT/WORD/2222/stage1/checkpoint-802.pth"
 export SUITE_PLAN='2222:2 2222:3 2222:4 pcdd_offline:1 pcdd_44:1 pcdd_44:2 pcdd_422:1 pcdd_422:2 pcdd_422:3 pcdd_71:1 pcdd_71:2'
elif [[ "$DATASET" == BTCV ]]; then
 export SUITE_PLAN='pcdd_44:1 pcdd_44:2 pcdd_422:1 pcdd_422:2 pcdd_422:3 pcdd_41111:1 pcdd_41111:2 pcdd_41111:3 pcdd_41111:4 pcdd_41111:5 pcdd_71:1 pcdd_71:2 pcdd_offline:1'
else
 exit 2
fi
printf '%s\n' "$SUITE_PLAN" > "$SUITE_ROOT/plan.txt"
export SUITE_INDEX=0
bash "$EXPERIMENT_WORKDIR/scripts/orgslot/pcdd_suite/queue.sh"
