#!/usr/bin/env bash
set -euo pipefail

REPO=/home/Anteng/OD_OWT_orgslot
TRAIN_SESSION=owt-word448-roi20
CHECKPOINT=${REPO}/Results/OWTLegacy/Common8/WORD_2D/OWT_WORD112_input448_roi20_legacyTG6_focusunion_eb192_u118800/checkpoint-802.pth
EVAL_LOG=${REPO}/Results/OWTLegacy/evaluation/Common8/WORD_2D/roi20_focusunion_ckpt802_fixedthr002.log

while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep 60
done

[[ -f "${CHECKPOINT}" ]] || { echo "training ended without final checkpoint: ${CHECKPOINT}" >&2; exit 2; }
mkdir -p "$(dirname "${EVAL_LOG}")"
CUDA_VISIBLE_DEVICES=0 bash "${REPO}/scripts/eval_word448_owt_roi20.sh" \
  2>&1 | tee "${EVAL_LOG}"
