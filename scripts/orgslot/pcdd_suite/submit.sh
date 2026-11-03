#!/usr/bin/env bash
set -euo pipefail
source "${EXPERIMENT_WORKDIR:?}/scripts/orgslot/pcdd_suite/environment.sh"
mkdir -p "$RUN_ROOT"
stages=$("$PYTHON" -c 'import sys;from util.orgslot_incremental2222 import protocol_groups;print(len(protocol_groups(sys.argv[1])))' "$PROTOCOL")
set -o noclobber
exec 3> "$RUN_ROOT/jobs.txt"
printf 'commit=%s runtime=%s dataset=%s protocol=%s account=sifansong qos=8gpus script=scripts/orgslot/pcdd_suite/train.sh\n' "$EXPECTED_COMMIT" "$EXPERIMENT_WORKDIR" "$DATASET" "$PROTOCOL" >&3
common=(--parsable --account=sifansong --qos=8gpus --nodes=1 --ntasks=1 --chdir="$EXPERIMENT_WORKDIR" --export=ALL --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err")
previous=${PREDECESSOR_JOB:-}
hold=(); [[ "${HOLD_FIRST:-0}" != 1 ]] || hold=(--hold)
for ((stage=${FIRST_STAGE:-1};stage<=stages;stage++)); do
 deps=(); [[ -z "$previous" ]] || deps=(--dependency="afterok:$previous")
 train=$(sbatch "${common[@]}" "${deps[@]}" "${hold[@]}" --partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=5-00:00:00 --job-name="${DATASET}_${PROTOCOL}_S$stage" scripts/orgslot/pcdd_suite/train.sh "$stage" initial)
 continuation=$(sbatch "${common[@]}" --dependency="afterany:$train" --partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=5-00:00:00 --job-name="${DATASET}_${PROTOCOL}_S${stage}_resume" scripts/orgslot/pcdd_suite/train.sh "$stage" resume)
 evaluation=$(sbatch "${common[@]}" --dependency="afterok:$continuation" --partition=gpu4090,gpu40901t --gres=gpu:4090:1 --cpus-per-task=10 --mem=128G --time=1-00:00:00 --job-name="${DATASET}_${PROTOCOL}_S${stage}_eval" scripts/orgslot/pcdd_suite/eval.sh "$stage")
 printf 'stage%s train=%s continuation=%s evaluation=%s predecessor=%s\n' "$stage" "$train" "$continuation" "$evaluation" "$previous" >&3
 # Evaluate every stage before advancing, so missing results stop the chain.
 previous=$evaluation
 hold=()
 for job in "$train" "$continuation" "$evaluation"; do scontrol show job "$job" -o; done
done
printf '%s\n' "$previous" > "$RUN_ROOT/last_job.txt"
