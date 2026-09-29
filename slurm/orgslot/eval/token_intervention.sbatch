#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_WORKDIR:?}" "${PYTHON:?}" "${CHECKPOINT:?}" "${DATA:?}" "${OUTPUT:?}"
cd "$EVAL_WORKDIR"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" -m tools.token_intervention --checkpoint "$CHECKPOINT" --train-csv "$DATA/csv/WORD_Training_2D_native07072.csv" --test-csv "$DATA/csv/WORD_Test_2D_native07072.csv" --output "$OUTPUT" --batch-size 2 --workers 4
