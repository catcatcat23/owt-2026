#!/usr/bin/env bash
set -euo pipefail
export EXPECTED_SLOT_HEAD_TYPE=arm_f_attention
exec bash "${EVAL_WORKDIR:?}/slurm/orgslot/eval/arm_e_mae_unified.sbatch"
