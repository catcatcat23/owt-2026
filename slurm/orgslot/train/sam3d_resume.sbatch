#!/usr/bin/env bash
# Explicit checkpoint, fresh Job-ID output; never overwrite the failed run.
set -euo pipefail
: "${RESUME_CHECKPOINT:?}" "${EXPERIMENT_WORKDIR:?}" "${ARM_F_TARGET:?}"
cd "$EXPERIMENT_WORKDIR"
export ARM_F_DIMENSION=3D MAE_INIT_CHECKPOINT= MAE_INIT_SCOPE=encoder
source scripts/orgslot/arm_f_environment.sh
"$PYTHON" - "$RESUME_CHECKPOINT" "$HSAM_SUPERVISION" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')
a = c['args']
assert a.slot_head_type == 'arm_f_sam_tail' and a.hsam_supervision == sys.argv[2]
assert a.batch_size == 6 and a.accum_iter == 2 and a.max_optimizer_updates == 118800
assert c['optimizer']['state'] and 'scaler' in c
assert all(torch.isfinite(t).all() for t in c['model'].values() if t.is_floating_point())
print('Verified resume epoch', c['epoch'], flush=True)
PY
provenance="$(dirname "$RESUME_CHECKPOINT")/mae_initial_checkpoint_load.json"
test -s "$provenance"
bash slurm/orgslot/train/arm_f_sam_tail_3d.sbatch
cp "$provenance" "$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_3D/ArmF_3D_${SLURM_JOB_ID}/mae_initial_checkpoint_load.json"
