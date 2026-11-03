#!/usr/bin/env bash
# Bounded queue: only one train/resume/eval triplet is submitted at a time.
set -euo pipefail
: "${SUITE_PLAN:?}" "${SUITE_ROOT:?}"
index=${SUITE_INDEX:-0}
read -r -a plan <<< "$SUITE_PLAN"
if (( index >= ${#plan[@]} )); then
 printf 'All planned stages evaluated successfully\n' > "$SUITE_ROOT/completed.txt"
 exit 0
fi
item=${plan[$index]}
export PROTOCOL=${item%:*} SUITE_INDEX=$index
stage=${item##*:}
source "${EXPERIMENT_WORKDIR:?}/scripts/orgslot/pcdd_suite/environment.sh"
mkdir -p "$RUN_ROOT" "$SUITE_ROOT/receipts"
# An existing receipt prevents accidental duplicate submission, even after a partial failure.
set -o noclobber
exec 3> "$SUITE_ROOT/receipts/$index.txt"
printf 'commit=%s runtime=%s dataset=%s protocol=%s stage=%s account=sifansong qos=8gpus predecessor=%s\n' "$EXPECTED_COMMIT" "$EXPERIMENT_WORKDIR" "$DATASET" "$PROTOCOL" "$stage" "${PREDECESSOR_JOB:-}" >&3
common=(--parsable --account=sifansong --qos=8gpus --nodes=1 --ntasks=1 --chdir="$EXPERIMENT_WORKDIR" --export=ALL --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err")
deps=(); [[ -z "${PREDECESSOR_JOB:-}" ]] || deps=(--dependency="afterok:$PREDECESSOR_JOB")
train=$(sbatch "${common[@]}" "${deps[@]}" --hold --partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=5-00:00:00 --job-name="${DATASET}_${PROTOCOL}_S$stage" scripts/orgslot/pcdd_suite/train.sh "$stage" initial)
printf 'train=%s\n' "$train" >&3
continuation=$(sbatch "${common[@]}" --dependency="afterany:$train" --partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=5-00:00:00 --job-name="${DATASET}_${PROTOCOL}_S${stage}_resume" scripts/orgslot/pcdd_suite/train.sh "$stage" resume)
printf 'continuation=%s\n' "$continuation" >&3
evaluation=$(sbatch "${common[@]}" --dependency="afterok:$continuation" --partition=gpu4090,gpu40901t --gres=gpu:4090:1 --cpus-per-task=10 --mem=128G --time=1-00:00:00 --job-name="${DATASET}_${PROTOCOL}_S${stage}_eval" scripts/orgslot/pcdd_suite/eval.sh "$stage")
printf 'evaluation=%s\n' "$evaluation" >&3
# Release only once the entire stage chain exists. Failed submission leaves train held.
if [[ "${HOLD_FIRST:-0}" != 1 ]]; then scontrol release "$train"; fi
for job in "$train" "$continuation" "$evaluation"; do scontrol show job "$job" -o; done
