#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
test "$(cat SOURCE_COMMIT)" = "${EXPECTED_COMMIT:?}"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=3D
source scripts/orgslot/arm_f_environment.sh
checkpoint=$("$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --final)
for mode in calibrate head reconstruction; do
 out="$RUN_ROOT/evaluation/$mode"
 if [[ -f "$out/results.json" ]]; then continue; fi
 csv="$BTCV_ROOT/csv/BTCV_Test_2D_native07072.csv"; count=6
 extra=()
 if [[ $mode == calibrate ]]; then csv="$BTCV_ROOT/csv/BTCV_Training_2D_native07072.csv"; count=24; fi
 if [[ $mode == head ]]; then extra=(--class-thresholds "$RUN_ROOT/evaluation/calibrate/selected_thresholds.json"); fi
 "$PYTHON" -m tools.eval_common8_orgslot_unified_3d --checkpoint "$checkpoint" --data-csv "$csv" --test-reference-csv "$BTCV_ROOT/csv/BTCV_Test_2D_native07072.csv" --preprocess-summary "$BTCV_ROOT/metadata/preprocess_summary.json" --output-dir "$out" --mode "$mode" --expected-cases "$count" --calibration-samples 5342 --input-size 512 --body-center --batch-size 1 --workers 4 "${extra[@]}"
done
