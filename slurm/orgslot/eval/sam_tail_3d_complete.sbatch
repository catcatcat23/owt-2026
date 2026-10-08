#!/usr/bin/env bash
set -euo pipefail
: "${TRAIN_JOB_ID:?}" "${EXPERIMENT_WORKDIR:?}" "${ARM_F_TARGET:?}"
cd "$EXPERIMENT_WORKDIR"
export ARM_F_DIMENSION=3D
source scripts/orgslot/arm_f_environment.sh
export PYTHONNOUSERSITE=1
train_dir="$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_3D/ArmF_3D_$TRAIN_JOB_ID"
CHECKPOINT=$("$PYTHON" - "$train_dir" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
r = json.loads((p / 'log.txt').read_text().splitlines()[-1])
assert r['train_optimizer_updates'] == 118800, r['train_optimizer_updates']
checkpoint = p / ('checkpoint-%s.pth' % r['epoch'])
assert checkpoint.is_file(), checkpoint
print(checkpoint)
PY
)
export CHECKPOINT
for EVAL_STAGE in calibrate head reconstruction; do
    export EVAL_STAGE
    bash slurm/orgslot/eval/unified_3d_protocol.sbatch
done
