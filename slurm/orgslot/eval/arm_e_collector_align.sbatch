#!/usr/bin/env bash
set -euo pipefail
cd "${EVAL_WORKDIR:?}"
export EXPECTED_SLOT_HEAD_TYPE=arm_e_multiscale_query EXPECTED_LAMBDA_SEG=0.01
"${PYTHON:?}" - "${EVAL_WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME:?}/checkpoint-802.pth" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')
assert c['args'].lambda_collector_attention == .01
assert not c['args'].mae_init_checkpoint
assert c['epoch'] == 802
print('Collector-alignment checkpoint identity verified')
PY
exec bash slurm/orgslot/eval/arm_e_mae_unified.sbatch
