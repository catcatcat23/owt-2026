#!/usr/bin/env bash
# Cluster/account/QoS/resources supplied explicitly by submission manifest.
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
export ARM_F_DIMENSION=2D ARM_F_HEAD_TYPE=arm_e_multiscale_query
export ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3
export ARM_F_LAMBDA_SEG=0.01 BACKGROUND_REDUCTION=topk
export RESUME_CHECKPOINT="" MAE_INIT_CHECKPOINT=""
source scripts/orgslot/arm_f_environment.sh
export LAMBDA_COLLECTOR_ATTENTION=0.01
export RUN_NAME=ArmE_CollectorAlign001_scratch_${SLURM_JOB_ID}
export OUTPUT_DIR=${EXPERIMENT_WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
printf 'Source=%s runtime=%s target=%s collector_lambda=%s\n' "${SOURCE_COMMIT:?}" "$EXPERIMENT_WORKDIR" "$ARM_F_TARGET" "$LAMBDA_COLLECTOR_ATTENTION"
test "$((N_GPU * MICRO_BATCH * ACCUM_ITER))" -eq 192
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT" --roi_index "$ROI_INDEX"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json, sys, torch
from pathlib import Path
root = Path(sys.argv[1])
record = json.loads((root / 'log.txt').read_text().splitlines()[-1])
assert record['train_optimizer_updates'] >= 118800, record
c = torch.load(root / 'checkpoint-802.pth', map_location='cpu')
a = c['args']
assert a.slot_head_type == 'arm_e_multiscale_query'
assert a.lambda_collector_attention == .01 and a.lambda_seg == .01
assert a.batch_size == 16 and a.accum_iter == 3
assert not a.resume and not a.mae_init_checkpoint
print('Collector alignment full-budget checkpoint verified')
PY
