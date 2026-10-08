#!/usr/bin/env bash
# Two separate runs: HSAM_SUPERVISION=soft_prior_aux / downsample_gt.
# Explicit account/resources/time and immutable runtime required at submission.
set -euo pipefail
case "${HSAM_SUPERVISION:?}" in soft_prior_aux|downsample_gt) ;; *) exit 2 ;; esac
export HSAM_SUPERVISION ARM_F_DIMENSION=3D ARM_F_HEAD_TYPE=arm_f_sam_tail
export BACKGROUND_REDUCTION=topk
# Scratch unless an explicit compatible MAE checkpoint is supplied. Do not
# silently substitute the historical 2D MAE checkpoint for a 3D encoder.
export MAE_INIT_SCOPE=${MAE_INIT_SCOPE:-encoder}
cd "${EXPERIMENT_WORKDIR:?}"
if [[ -e .git ]]; then
    test "$(git rev-parse HEAD)" = "${ARM_F_COMMIT:?}"
else
    test "$(<SOURCE_COMMIT)" = "${ARM_F_COMMIT:?}"
fi
bash slurm/orgslot/train/arm_f.sbatch
