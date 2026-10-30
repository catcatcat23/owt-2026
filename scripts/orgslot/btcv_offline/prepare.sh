#!/usr/bin/env bash
# Run on the host containing raw BTCV; never reuse WORD ROI indices.
set -euo pipefail
: "${PYTHON:?}" "${BTCV_SOURCE_ROOT:?}" "${BTCV_SPLIT_JSON:?}" "${BTCV_OUTPUT_ROOT:?}"
"$PYTHON" tools/preprocess_common8_112_448.py --dataset BTCV \
 --source-root "$BTCV_SOURCE_ROOT" --split-json "$BTCV_SPLIT_JSON" \
 --output-root "$BTCV_OUTPUT_ROOT" --spacing 0.7 0.7 2 --manifest-tag native07072 \
 --runtime-input-size 448 --image-order 3 --workers "${PREPROCESS_WORKERS:-2}"
"$PYTHON" tools/audit_btcv_fixed_crop.py "$BTCV_OUTPUT_ROOT/BTCV"
"$PYTHON" tools/build_small_organ_roi_index.py \
 --manifest "$BTCV_OUTPUT_ROOT/BTCV/csv/BTCV_Training_2D_native07072.csv" \
 --output "$BTCV_OUTPUT_ROOT/BTCV/metadata/small_organ_roi_index.json"
