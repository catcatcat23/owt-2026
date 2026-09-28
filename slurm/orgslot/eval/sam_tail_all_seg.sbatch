#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_WORKDIR:?}" "${RUN_NAME:?}" "${PYTHON:?}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" - "$EVAL_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME/checkpoint-802.pth" <<'PYCODE'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
assert c['args'].seg_supervision == 'all'
print('Verified all-slot segmentation supervision')
PYCODE
export HSAM_SUPERVISION=none
exec bash "$EVAL_WORKDIR/slurm/orgslot/eval/sam_tail_guided.sbatch"
