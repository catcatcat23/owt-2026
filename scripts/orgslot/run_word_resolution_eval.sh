#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-/home/Anteng/miniconda3/envs/abdpet/bin/python}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT}
TEST_CSV=${TEST_CSV:?Set TEST_CSV}
PREPROCESS_SUMMARY=${PREPROCESS_SUMMARY:?Set PREPROCESS_SUMMARY}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
INPUT_SIZE=${INPUT_SIZE:?Set INPUT_SIZE}
GLOBAL_CROP_SIZE=${GLOBAL_CROP_SIZE:?Set GLOBAL_CROP_SIZE}
EXPECTED_SPACING=${EXPECTED_SPACING:?Set EXPECTED_SPACING as 'sx sy sz'}
EXPECTED_CASES=${EXPECTED_CASES:-24}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-8}
GPU_ID=${GPU_ID:-}

for required in "${PYTHON}" "${CHECKPOINT}" "${TEST_CSV}" "${PREPROCESS_SUMMARY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
read -r -a SPACING_ARRAY <<< "${EXPECTED_SPACING}"
if [[ "${#SPACING_ARRAY[@]}" -ne 3 ]]; then
  echo "EXPECTED_SPACING needs exactly three values" >&2
  exit 2
fi
if [[ -n "${GPU_ID}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
"${PYTHON}" -c "import os, torch; count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch_cuda=', torch.__version__, torch.version.cuda, 'count=', count); assert torch.cuda.is_available() and count == 1; torch.ones(1, device='cuda')"

cd "${REPO_ROOT}"
"${PYTHON}" tools/eval_common8_orgslot_reconstruction_threshold.py   --method orgslot   --checkpoint "${CHECKPOINT}"   --test-csv "${TEST_CSV}"   --preprocess-summary "${PREPROCESS_SUMMARY}"   --expected-spacing "${SPACING_ARRAY[@]}"   --output-dir "${OUTPUT_DIR}"   --class-map configs/orgslot/common8_offline.json   --input-size "${INPUT_SIZE}"   --global-crop-size "${GLOBAL_CROP_SIZE}"   --batch-size "${BATCH_SIZE}"   --workers "${WORKERS}"   --threshold 0.02   --min-size 20   --opening-radius 1   --class-ids 1,2,3,4,5,6,7,8   --device cuda   --fusion-mode linear_sqrt

"${PYTHON}" tools/validate_word_orgslot_resolution_eval.py   --eval-dir "${OUTPUT_DIR}"   --expected-input-size "${INPUT_SIZE}"   --expected-global-crop "${GLOBAL_CROP_SIZE}"   --expected-spacing "${SPACING_ARRAY[@]}"   --expected-cases "${EXPECTED_CASES}"
