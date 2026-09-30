#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_WORKDIR:?}" "${PYTHON:?}" "${CHECKPOINT:?}" "${DATA:?}" "${OUTPUT:?}"
cd "$EVAL_WORKDIR"
export PYTHONPATH="$EVAL_WORKDIR" PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
for cid in 4 6; do
  "$PYTHON" -m tools.frozen_p4_probe --checkpoint "$CHECKPOINT" --train-csv "$DATA/csv/WORD_Training_2D_native07072.csv" --test-csv "$DATA/csv/WORD_Test_2D_native07072.csv" --class-id "$cid" --epochs 10 --batch-size 16 --workers 4 --output "${OUTPUT}_class${cid}"
done
