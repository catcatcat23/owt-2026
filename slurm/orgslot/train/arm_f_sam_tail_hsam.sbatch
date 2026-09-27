#!/usr/bin/env bash
# Submit twice with HSAM_SUPERVISION=downsample_gt / upsample_logits.
# All resource/account options must be supplied explicitly at submission.
set -euo pipefail
case "${HSAM_SUPERVISION:?}" in downsample_gt|upsample_logits|m2f_hard) ;; *) exit 2 ;; esac
export HSAM_SUPERVISION
cd "${EXPERIMENT_WORKDIR:?}"
bash slurm/orgslot/train/arm_f_sam_tail_encoder.sbatch
