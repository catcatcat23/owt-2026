#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
DATA_ROOT=${DATA_ROOT:-/mnt/DATA-4/anteng}
SOURCE_ROOT=${SOURCE_ROOT:-${DATA_ROOT}/WORD/raw/WORD-V0.1.0}
SPLIT_JSON=${SPLIT_JSON:-${DATA_ROOT}/processed/OWT_Common8/WORD/metadata/WORD_split_seed42.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/processed/OWT_Common8_112_NATIVE}
TRAIN_CSV=${OUTPUT_ROOT}/WORD/csv/WORD_Training_2D_native112.csv
ROI_INDEX=${OUTPUT_ROOT}/WORD/metadata/small_organ_roi_index.json
ROI_AUDIT=${OUTPUT_ROOT}/WORD/metadata/small_organ_roi_audit.json

if [[ ! -x "${PYTHON}" || ! -d "${SOURCE_ROOT}" || ! -f "${SPLIT_JSON}" ]]; then
  echo "Missing Python, WORD source root, or split JSON" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}/WORD" ]]; then
  echo "Refusing to overwrite ${OUTPUT_ROOT}/WORD" >&2
  exit 2
fi

cd "${REPO_ROOT}"
"${PYTHON}" tools/preprocess_common8_112_448.py \
  --dataset WORD \
  --source-root "${SOURCE_ROOT}" \
  --split-json "${SPLIT_JSON}" \
  --output-root "${OUTPUT_ROOT}" \
  --spacing 1 1 2 \
  --hu-clip -175 250 \
  --image-order 3 \
  --jpeg-quality 95

"${PYTHON}" tools/build_small_organ_roi_index.py \
  --manifest "${TRAIN_CSV}" \
  --output "${ROI_INDEX}" \
  --focus-class-ids 4,5,6

"${PYTHON}" tools/audit_small_organ_roi_policy.py \
  --manifest "${TRAIN_CSV}" \
  --roi-index "${ROI_INDEX}" \
  --output "${ROI_AUDIT}" \
  --trials-per-slice 4 \
  --seed 0

echo "WORD native 1x1x2 preprocessing complete: ${OUTPUT_ROOT}/WORD"
