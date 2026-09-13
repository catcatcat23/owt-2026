#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${MAE_INIT_SCOPE:?}" "${MAE_INIT_CHECKPOINT:?}"
case "$MAE_INIT_SCOPE" in encoder|encoder_decoder) ;; *) exit 2 ;; esac
export ARM_F_DIMENSION=2D BACKGROUND_REDUCTION=topk RESUME_CHECKPOINT=""
cd "$EXPERIMENT_WORKDIR"
printf '0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  %s\n' "$MAE_INIT_CHECKPOINT" | sha256sum -c -
bash slurm/orgslot/train/arm_f.sbatch
test -s "Results/OrganSlotBank/Common8/WORD_2D/ArmF_2D_${SLURM_JOB_ID}/mae_initial_checkpoint_load.json"
