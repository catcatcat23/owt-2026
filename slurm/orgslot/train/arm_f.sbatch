#!/usr/bin/env bash
# Resource options are supplied explicitly at sbatch for the selected cluster.
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
source scripts/orgslot/arm_f_environment.sh
export RUN_NAME=ArmF_${DIMENSION}_${SLURM_JOB_ID}
export OUTPUT_DIR=${EXPERIMENT_WORKDIR}/Results/OrganSlotBank/Common8/WORD_${DIMENSION}/${RUN_NAME}
printf 'Pinned commit=%s runtime=%s target=%s dimension=%s\n' "${ARM_F_COMMIT:?}" "$EXPERIMENT_WORKDIR" "$ARM_F_TARGET" "$DIMENSION"
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT"
test -r "$ROI_INDEX"
test -r "$LPIPS_STATE"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -s "${OUTPUT_DIR}/log.txt"
"$PYTHON" -c 'import json,sys; from pathlib import Path; r=json.loads(Path(sys.argv[1]).read_text().splitlines()[-1]); assert r["train_optimizer_updates"] >= 118800, r; print("Full update budget verified")' "${OUTPUT_DIR}/log.txt"
