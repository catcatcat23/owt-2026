#!/usr/bin/env bash
set -euo pipefail
export MAE_RESUME_SCOPE=encoder
exec bash "${EXPERIMENT_WORKDIR:?}/slurm/orgslot/train/arm_e_mae_encdec_resume500.sbatch"
