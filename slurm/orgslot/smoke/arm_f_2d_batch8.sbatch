#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
source scripts/orgslot/arm_f_environment.sh
export OMP_NUM_THREADS=2
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 -m tools.smoke_arm_f_2d_batch8
