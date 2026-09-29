#!/usr/bin/env bash
set -euo pipefail
case "${HSAM_SUPERVISION:?}" in soft_prior|soft_prior_aux) ;; *) exit 2 ;; esac
export ARM_F_SEG_SUPERVISION=retained
cd "${EXPERIMENT_WORKDIR:?}"
exec bash slurm/orgslot/train/arm_f_sam_tail_encoder.sbatch
