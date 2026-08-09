#!/usr/bin/env bash
set -euo pipefail

REPO=/home/Anteng/OD_OWT_orgslot
PYTHON=/home/Anteng/miniconda3/envs/abdpet/bin/python
CHECKPOINT=${REPO}/Results/OWTLegacy/Common8/WORD_2D/OWT_WORD112_input448_roi20_legacyTG6_focusunion_eb192_u118800/checkpoint-802.pth
OUTPUT=${REPO}/Results/OWTLegacy/evaluation/Common8/WORD_2D/roi20_focusunion_ckpt802_fixedthr002
GPU_LIST=${CUDA_VISIBLE_DEVICES:-0}

cd "${REPO}"
[[ -f "${CHECKPOINT}" ]] || { echo "missing final checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "evaluation output already exists: ${OUTPUT}" >&2; exit 3; }

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON}" \
  -m tools.eval_common8_orgslot_reconstruction_threshold \
  --method owt --checkpoint "${CHECKPOINT}" \
  --test-csv /mnt/DATA-4/anteng/processed/OWT_Common8_112_NATIVE/WORD/csv/WORD_Test_2D_native112.csv \
  --output-dir "${OUTPUT}" \
  --class-map configs/orgslot/common8_offline.json \
  --input-size 448 --global-crop-size 448 \
  --batch-size 4 --workers 8 --threshold 0.02 \
  --min-size 20 --opening-radius 1 \
  --class-ids 1,2,3,4,5,6,7,8 \
  --print-freq 20 --device cuda
