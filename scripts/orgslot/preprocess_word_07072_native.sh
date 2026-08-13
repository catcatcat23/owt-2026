#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python}
DATA_ROOT=${DATA_ROOT:-/gpfs/work/aac/bolinren19/2026-07/DATA}
SOURCE_ROOT=${SOURCE_ROOT:-${DATA_ROOT}/WORD/raw/WORD-V0.1.0}
SPLIT_JSON=${SPLIT_JSON:-${DATA_ROOT}/processed/OWT_Common8/WORD/metadata/WORD_split_seed42.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/processed/OWT_Common8_07072_NATIVE}
DATASET_ROOT=${OUTPUT_ROOT}/WORD
TRAIN_CSV=${DATASET_ROOT}/csv/WORD_Training_2D_native07072.csv
TEST_CSV=${DATASET_ROOT}/csv/WORD_Test_2D_native07072.csv
ROI_INDEX=${DATASET_ROOT}/metadata/small_organ_roi_index.json
ROI_AUDIT=${DATASET_ROOT}/metadata/small_organ_roi_audit.json
CROP_AUDIT=${DATASET_ROOT}/metadata/crop448_roi384_audit.json
TRAIN_CROP_AUDIT=${DATASET_ROOT}/metadata/train_crop448_roi384_audit.json
PREPROCESS_WORKERS=${PREPROCESS_WORKERS:-3}
REUSE_GEOMETRY_PREFLIGHT=${REUSE_GEOMETRY_PREFLIGHT:-0}
REUSE_COMPLETE_CASES=${REUSE_COMPLETE_CASES:-0}

for required in "${PYTHON}" "${SOURCE_ROOT}" "${SPLIT_JSON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${DATASET_ROOT}" && "${REUSE_GEOMETRY_PREFLIGHT}" != "1" ]]; then
  echo "Refusing to overwrite ${DATASET_ROOT}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
PREPROCESS_ARGS=(
  --dataset WORD
  --source-root "${SOURCE_ROOT}"
  --split-json "${SPLIT_JSON}"
  --output-root "${OUTPUT_ROOT}"
  --spacing 0.7 0.7 2
  --runtime-input-size 448
  --manifest-tag native07072
  --hu-clip -175 250
  --image-order 3
  --jpeg-quality 95
  --workers "${PREPROCESS_WORKERS}"
)
if [[ "${REUSE_GEOMETRY_PREFLIGHT}" == "1" ]]; then
  PREPROCESS_ARGS+=(--reuse-geometry-preflight)
fi
"${PYTHON}" tools/preprocess_common8_112_448.py "${PREPROCESS_ARGS[@]}"

"${PYTHON}" tools/build_small_organ_roi_index.py   --manifest "${TRAIN_CSV}"   --output "${ROI_INDEX}"   --focus-class-ids 4,5,6
if [[ "${REUSE_COMPLETE_CASES}" == "1" ]]; then
  PREPROCESS_ARGS+=(--reuse-complete-cases)
fi

"${PYTHON}" tools/audit_small_organ_roi_policy.py   --manifest "${TRAIN_CSV}"   --roi-index "${ROI_INDEX}"   --output "${ROI_AUDIT}"   --trials-per-slice 4   --seed 0

"${PYTHON}" tools/analyze_word_crop_geometry.py   --csv "${TRAIN_CSV}" "${TEST_CSV}"   --crop-sizes 448   --roi-size 384   --output "${CROP_AUDIT}"

"${PYTHON}" tools/analyze_word_crop_geometry.py   --csv "${TRAIN_CSV}"   --crop-sizes 448   --roi-size 384   --output "${TRAIN_CROP_AUDIT}"

echo "WORD native 0.7x0.7x2 preprocessing complete: ${DATASET_ROOT}"
