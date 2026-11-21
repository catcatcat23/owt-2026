#!/usr/bin/env bash
# CPU-only completion check: do not queue another GPU just to discover training finished.
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
test "$(cat SOURCE_COMMIT)" = "${EXPECTED_COMMIT:?}"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=3D
source scripts/orgslot/arm_f_environment.sh
if "$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --final > "$RUN_ROOT/final_checkpoint.txt.tmp" 2> "$RUN_ROOT/chain_check.log"; then
 mv "$RUN_ROOT/final_checkpoint.txt.tmp" "$RUN_ROOT/final_checkpoint.txt"
 job=$(sbatch --parsable --export=ALL --account=sifansong --qos=4gpus --partition=gpu4090 --gres=gpu:4090:1 --cpus-per-task=4 --mem=96G --time=05:00:00 --dependency=afterok:${SLURM_JOB_ID:?} --job-name=BTCV3D_eval --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err" scripts/orgslot/btcv_3d/eval.sh)
 printf 'evaluation %s\n' "$job" >> "$RUN_ROOT/job_chain.txt"
else
 # Require a valid recovery checkpoint before allocating another GPU job.
 "$PYTHON" tools/btcv_offline_checkpoint.py "$RUN_ROOT" --resume > "$RUN_ROOT/resume_checkpoint.txt"
 checkpoint=$(cat "$RUN_ROOT/resume_checkpoint.txt")
 [[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "No recovery checkpoint; stop chain without allocating GPUs"; exit 2; }
 [[ ${CHAIN_ATTEMPT:-1} -lt 4 ]] || { echo 'Four allocations exhausted; inspect before continuing'; exit 2; }
 export CHAIN_ATTEMPT=$(( ${CHAIN_ATTEMPT:-1} + 1 ))
 job=$(sbatch --parsable --export=ALL --account=sifansong --qos=4gpus --partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=7-00:00:00 --job-name=BTCV3D_resume --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err" scripts/orgslot/btcv_3d/train.sh resume)
 printf 'resume %s\n' "$job" >> "$RUN_ROOT/job_chain.txt"
 check=$(sbatch --parsable --export=ALL --account=sifansong --qos=4gpus --partition=gpua800,gpua8001t --cpus-per-task=2 --mem=12G --time=00:10:00 --dependency=afterany:$job --job-name=BTCV3D_chain --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err" scripts/orgslot/btcv_3d/chain.sh)
 printf 'check %s\n' "$check" >> "$RUN_ROOT/job_chain.txt"
fi
