#!/usr/bin/env bash
set -euo pipefail
: "${HSAM_SUPERVISION:?}" "${EVAL_WORKDIR:?}" "${RUN_NAME:?}" "${PYTHON:?}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" - "$EVAL_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME/checkpoint-802.pth" "$HSAM_SUPERVISION" <<'PYCODE'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
assert c['args'].hsam_supervision == sys.argv[2]
assert c['args'].mae_init_scope == 'encoder'
print('Verified guided SAM-tail configuration:', sys.argv[2])
PYCODE
export EXPECTED_SLOT_HEAD_TYPE=arm_f_sam_tail
exec bash "$EVAL_WORKDIR/slurm/orgslot/eval/arm_e_mae_unified.sbatch"
