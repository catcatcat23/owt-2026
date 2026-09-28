#!/usr/bin/env bash
# Resource/account/time options supplied at submission; separate ablation.
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}"
export ARM_F_SEG_SUPERVISION=all
export HSAM_SUPERVISION=none
cd "$EXPERIMENT_WORKDIR"
bash slurm/orgslot/train/arm_f_sam_tail_encoder.sbatch
