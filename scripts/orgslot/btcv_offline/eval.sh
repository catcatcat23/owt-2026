#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
test "$(cat SOURCE_COMMIT)" = "${EXPECTED_COMMIT:?}"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
checkpoint=$("$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --final)
output=$RUN_ROOT/evaluation
common=(--input-size 512 --global-crop-size 512 --body-center --checkpoint "$checkpoint" --batch-size 4 --workers 4 --min-size 20 --opening-radius 1 --device cuda)
"$PYTHON" -m tools.eval_common8_orgslot_heads "${common[@]}" --data-csv "$BTCV_ROOT/csv/BTCV_Training_2D_native07072.csv" --output-dir "$output/train_calibration" --split train_calibration --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
"$PYTHON" -m tools.eval_common8_orgslot_heads "${common[@]}" --data-csv "$BTCV_ROOT/csv/BTCV_Test_2D_native07072.csv" --output-dir "$output/test" --split test --binary-threshold 0.5 --class-thresholds "$output/train_calibration/selected_thresholds.json"
"$PYTHON" -m tools.eval_common8_orgslot_reconstruction_threshold --method orgslot --checkpoint "$checkpoint" --test-csv "$BTCV_ROOT/csv/BTCV_Test_2D_native07072.csv" --preprocess-summary "$BTCV_ROOT/metadata/preprocess_summary.json" --expected-spacing 0.7 0.7 2 --output-dir "$output/reconstruction" --input-size 512 --global-crop-size 512 --body-center --batch-size 4 --workers 4 --threshold 0.02 --min-size 20 --opening-radius 1 --device cuda
