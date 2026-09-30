#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${RESUME_CHECKPOINT:?}" "${ARM_F_COMMIT:?}"
cd "$EXPERIMENT_WORKDIR"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=2D ARM_F_HEAD_TYPE=arm_f_sam_tail
export ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3 ARM_F_LAMBDA_SEG=0.01
export ARM_F_SEG_SUPERVISION=retained BACKGROUND_REDUCTION=topk
export HSAM_SUPERVISION=soft_prior_aux MAE_INIT_SCOPE=encoder MAE_INIT_CHECKPOINT=""
source scripts/orgslot/arm_f_environment.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
# Actual NCCL schema regression on two of the allocated GPUs, before training.
METRICS_TEST_BACKEND=nccl PYTHONPATH=.:tests "$PYTHON" -m unittest test_distributed_metrics -v
"$PYTHON" - "$RESUME_CHECKPOINT" <<'PY_CHECK'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')
a = c['args']
assert c['epoch'] == 0 and a.hsam_supervision == 'soft_prior_aux'
assert a.slot_head_type == 'arm_f_sam_tail' and a.mae_init_scope == 'encoder'
assert a.batch_size == 16 and a.accum_iter == 3 and a.lambda_seg == .01
assert c['optimizer']['state'] and 'scaler' in c
assert all(torch.isfinite(t).all() for t in c['model'].values() if t.is_floating_point())
print('Resume checkpoint verified; start_epoch=1; optimizer/scaler restored by trainer', flush=True)
PY_CHECK
output="$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/ArmF_2D_${SLURM_JOB_ID}"
mkdir -p "$output"
cp "$(dirname "$RESUME_CHECKPOINT")/mae_initial_checkpoint_load.json" "$output/mae_initial_checkpoint_load.json"
exec bash slurm/orgslot/train/arm_f.sbatch
