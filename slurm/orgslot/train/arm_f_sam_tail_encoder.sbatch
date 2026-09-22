#!/usr/bin/env bash
# Resource/account/time options are supplied at submission.
set -euo pipefail
export ARM_F_HEAD_TYPE=arm_f_sam_tail
export ARM_F_DIMENSION=2D MAE_INIT_SCOPE=encoder
export ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3 ARM_F_LAMBDA_SEG=0.01
export BACKGROUND_REDUCTION=topk RESUME_CHECKPOINT=""
cd "${EXPERIMENT_WORKDIR:?}"
bash slurm/orgslot/train/arm_f_mae.sbatch
