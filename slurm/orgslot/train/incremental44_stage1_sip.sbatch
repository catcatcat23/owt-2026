#!/usr/bin/env bash
#SBATCH --job-name=Inc44_Ecross_MAE_stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${NEW_DATA_ROOT:?}" "${RUN_ROOT:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=normal)"
export ARM_F_TARGET=bolin-sip ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
export PYTHON_BIN="$PYTHON" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export WORD_ROOT="$NEW_DATA_ROOT/WORD"
export MAE_CHECKPOINT=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_autopet_mae/Results/AutoPET_MAE/MAE_LA6D8_224_M75_U100K_seed0/checkpoint-final.pth
test -s "$WORD_ROOT/metadata/READY.json"
test -s "$LPIPS_STATE"
printf '0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  %s\n' "$MAE_CHECKPOINT" | sha256sum -c -
"$PYTHON_BIN" tools/prepare_word_official96.py validate --plan configs/orgslot/word_official96_seed42.json --output "$NEW_DATA_ROOT"
mkdir -p "$RUN_ROOT"
# Two full-batch optimizer updates exercise the actual five-slot DDP path.
# A failure stops this job; formal training always starts fresh from MAE.
export OUTPUT_DIR="$RUN_ROOT/preflight_${SLURM_JOB_ID}"
bash scripts/orgslot/run_incremental44.sh stage1 --max_optimizer_updates 2 --warmup_updates 0 --no_save
"$PYTHON_BIN" - "$OUTPUT_DIR/log.txt" <<'PY_CHECK'
import json, math, sys
record = json.loads(open(sys.argv[1]).read().splitlines()[-1])
assert record['optimizer_updates'] == 2, record
assert all(not isinstance(v, (int, float)) or math.isfinite(v) for v in record.values()), record
print('GPU_DDP_PREFLIGHT_PASSED: two updates, finite logged metrics', flush=True)
PY_CHECK
export OUTPUT_DIR="$RUN_ROOT/stage1_${SLURM_JOB_ID}"
bash scripts/orgslot/run_incremental44.sh stage1 --save_freq 10
