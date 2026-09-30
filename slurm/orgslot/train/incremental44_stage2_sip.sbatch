#!/usr/bin/env bash
#SBATCH --job-name=Inc44_Ecross_MAE_stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${NEW_DATA_ROOT:?}" "${RUN_ROOT:?}" "${STAGE1_RUN:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"
export ARM_F_TARGET=bolin-sip ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export PYTHON_BIN="$PYTHON" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export WORD_ROOT="$NEW_DATA_ROOT/WORD"
export BACKGROUND_POLICY=${BACKGROUND_POLICY:-separation}
test -s "$WORD_ROOT/metadata/READY.json"
"$PYTHON_BIN" tools/prepare_word_official96.py validate --plan configs/orgslot/word_official96_seed42.json --output "$NEW_DATA_ROOT"
STAGE1_CHECKPOINT=$("$PYTHON_BIN" tools/incremental44_final_checkpoint.py "$STAGE1_RUN")
export STAGE1_CHECKPOINT
sha256sum "$STAGE1_CHECKPOINT"
mkdir -p "$RUN_ROOT"
export OUTPUT_DIR="$RUN_ROOT/stage2_preflight_${SLURM_JOB_ID}"
bash scripts/orgslot/run_incremental44.sh stage2 --max_optimizer_updates 2 --warmup_updates 0 --no_save
"$PYTHON_BIN" - "$OUTPUT_DIR/log.txt" <<'PY_CHECK'
import json, math, sys
record = json.loads(open(sys.argv[1]).read().splitlines()[-1])
assert record['optimizer_updates'] == 2, record
assert all(not isinstance(v, (int, float)) or math.isfinite(v) for v in record.values()), record
print('STAGE2_GPU_DDP_PREFLIGHT_PASSED', flush=True)
PY_CHECK
# Discard the preflight weights and initialize again from the exact Stage1.
export OUTPUT_DIR="$RUN_ROOT/stage2_${BACKGROUND_POLICY}_${SLURM_JOB_ID}"
bash scripts/orgslot/run_incremental44.sh stage2 --save_freq 10
