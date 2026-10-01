#!/usr/bin/env bash
# Resource/account options are supplied at submission.
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${PYTHON:?}"
: "${OLD_DATA:?}" "${OLD_RUN:?}" "${OLD_RESULTS:?}" "${THRESHOLDS:?}"
: "${REPLACEMENT_DATA:?}" "${OUTPUT_ROOT:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
COMMON=(--plan configs/orgslot/word_replacement5_seed42.json
 --old-data "$OLD_DATA" --old-run "$OLD_RUN" --old-results "$OLD_RESULTS"
 --replacement-data "$REPLACEMENT_DATA" --output "$OUTPUT_ROOT" --thresholds "$THRESHOLDS")
"$PYTHON" tools/word_replacement_eval.py prepare "${COMMON[@]}"
"$PYTHON" - "$OLD_RUN/checkpoint-802.pth" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')
a = c['args']
assert c['epoch'] == 802 and a.slot_head_type == 'arm_f_reverse_dot'
assert a.lambda_seg == 0.01 and not a.mae_init_checkpoint
assert a.input_size == 448 and list(a.expected_spacing) == [0.7, 0.7, 2.0]
print('Verified F Reverse-Dot scratch lambda_seg=0.01 checkpoint-802')
PY
"$PYTHON" -m tools.eval_common8_orgslot_heads \
 --checkpoint "$OLD_RUN/checkpoint-802.pth" --data-csv "$OUTPUT_ROOT/test.csv" \
 --output-dir "$OUTPUT_ROOT/heads" --class-map configs/orgslot/common8_offline.json \
 --split test --binary-threshold 0.5 --class-thresholds "$THRESHOLDS" \
 --fusion-mode linear_sqrt --batch-size 4 --workers 4 --min-size 20 --opening-radius 1 --device cuda
"$PYTHON" tools/word_replacement_eval.py compare "${COMMON[@]}"
