#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${SUITE_ROOT:?}" "${DATA_ROOT:?}" "${DATASET:?}" "${PROTOCOL:?}" "${ARM_F_TARGET:?}" "${MAE_CHECKPOINT:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(cat SOURCE_COMMIT)" = "$EXPECTED_COMMIT"
export ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
RUN_ROOT=$SUITE_ROOT/$DATASET/$PROTOCOL
TRAIN_CSV=$DATA_ROOT/csv/${DATASET}_Training_2D_native07072.csv
TEST_CSV=$DATA_ROOT/csv/${DATASET}_Test_2D_native07072.csv
VAL_CSV=$DATA_ROOT/csv/${DATASET}_Validation_2D_native07072.csv
GEOMETRY=()
TRAIN_GEOMETRY=()
if [[ "$DATASET" == BTCV ]]; then
 VAL_CSV=$TRAIN_CSV
 GEOMETRY=(--input-size 512 --global-crop-size 512 --body-center)
 TRAIN_GEOMETRY=(--input_size 512 --global_crop_size 512 --body_center --no_validation)
else
 [[ "$DATASET" == WORD ]] || exit 2
fi
CONFIG=configs/orgslot/$PROTOCOL.json
[[ "$PROTOCOL" != 2222 ]] || CONFIG=configs/orgslot/common8_incremental2222.json
