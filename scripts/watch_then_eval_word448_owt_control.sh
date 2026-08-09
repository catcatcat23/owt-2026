#!/usr/bin/env bash
set -euo pipefail

REPO=/home/Anteng/OD_OWT_orgslot
TRAIN_SESSION=owt-word448-control
CHECKPOINT=${REPO}/Results/OWTLegacy/Common8/WORD_2D/OWT_WORD112_input448_control_legacyTG6_eb192_u118800/checkpoint-802.pth
EVAL_LOG=${REPO}/Results/OWTLegacy/evaluation/Common8/WORD_2D/control_ckpt802_fixedthr002.log

while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "training session ended without final checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${EVAL_LOG}")"
CUDA_VISIBLE_DEVICES=0 bash "${REPO}/scripts/eval_word448_owt_control.sh" \
  2>&1 | tee "${EVAL_LOG}"
