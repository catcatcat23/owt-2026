#!/usr/bin/env bash
# Resource options are supplied explicitly at sbatch for the selected cluster.
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
source scripts/orgslot/arm_f_environment.sh
export RUN_NAME=ArmF_${DIMENSION}_${SLURM_JOB_ID}
export OUTPUT_DIR=${EXPERIMENT_WORKDIR}/Results/OrganSlotBank/Common8/WORD_${DIMENSION}/${RUN_NAME}
printf 'Pinned commit=%s runtime=%s target=%s dimension=%s\n' "${ARM_F_COMMIT:?}" "$EXPERIMENT_WORKDIR" "$ARM_F_TARGET" "$DIMENSION"
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT" --roi_index "$ROI_INDEX"
test -r "$ROI_INDEX"
test -r "$LPIPS_STATE"
case "${HSAM_SUPERVISION:-none}" in
  downsample_gt_soft_prior|downsample_gt_soft_prior_aux)
    OUTPUT_DIR="${OUTPUT_DIR}_preflight" MAX_UPDATES=2 WARMUP_UPDATES=0 NO_SAVE=1 \
      bash scripts/orgslot/run_word_112_448_a100.sh roi20
    "$PYTHON" - "${OUTPUT_DIR}_preflight/log.txt" <<'PY_PREFLIGHT'
import json, math, sys
with open(sys.argv[1]) as handle:
    row = json.loads(handle.read().splitlines()[-1])
assert row['train_optimizer_updates'] == 2, row
assert all(not isinstance(v, (float, int)) or math.isfinite(v) for v in row.values())
print('GPU_DDP_PREFLIGHT_PASSED: combined coarse/soft prior, two updates')
PY_PREFLIGHT
    ;;
esac
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -s "${OUTPUT_DIR}/log.txt"
"$PYTHON" -c 'import json,sys; from pathlib import Path; r=json.loads(Path(sys.argv[1]).read_text().splitlines()[-1]); assert r["train_optimizer_updates"] >= 118800, r; print("Full update budget verified")' "${OUTPUT_DIR}/log.txt"
