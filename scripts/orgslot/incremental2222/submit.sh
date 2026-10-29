#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${RUN_ROOT:?}" "${WORD_ROOT:?}" "${MAE_CHECKPOINT:?}"
[[ "$(id -un)" == antengcai23 ]] || { echo 'Run on antengcai23/XEC'; exit 2; }
cd "$EXPERIMENT_WORKDIR"
test "$(cat SOURCE_COMMIT)" = "$EXPECTED_COMMIT"
test -s "$WORD_ROOT/metadata/READY.json"
test -s "$MAE_CHECKPOINT"
export ARM_F_TARGET=anteng-xec ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
"$PYTHON" tools/prepare_word_official96.py validate --plan configs/orgslot/word_official96_seed42.json --output "$(dirname "$WORD_ROOT")"
mkdir -p "$RUN_ROOT"
set -o noclobber
exec 3> "$RUN_ROOT/jobs.txt"
printf 'commit=%s runtime=%s account=sifansong user=antengcai23 cluster=XEC qos=8gpus\n' "$EXPECTED_COMMIT" "$EXPERIMENT_WORKDIR" >&3
common=(--parsable --account=sifansong --qos=8gpus --nodes=1 --ntasks=1 --chdir="$EXPERIMENT_WORKDIR" --export=ALL --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err")
previous=''
for stage in 1 2 3 4; do
 dependency=()
 [[ -z "$previous" ]] || dependency=(--dependency="afterok:$previous")
 train=$(sbatch "${common[@]}" "${dependency[@]}" --partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=5-00:00:00 --job-name="SAM2222_S${stage}" scripts/orgslot/incremental2222/train.sh "$stage")
 printf 'stage%s_train=%s script=scripts/orgslot/incremental2222/train.sh\n' "$stage" "$train" >&3
 # A second allocation continues the last valid checkpoint after timeout. No scratch fallback.
 continuation=$(sbatch "${common[@]}" --dependency="afterany:$train" --partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=5-00:00:00 --job-name="SAM2222_S${stage}_resume" scripts/orgslot/incremental2222/train.sh "$stage")
 printf 'stage%s_continuation=%s afterany=%s\n' "$stage" "$continuation" "$train" >&3
 evaluation=$(sbatch "${common[@]}" --dependency="afterok:$continuation" --partition=gpu4090,gpu40901t --gres=gpu:4090:1 --cpus-per-task=10 --mem=128G --time=1-00:00:00 --job-name="SAM2222_S${stage}_eval" scripts/orgslot/incremental2222/eval.sh "$stage")
 printf 'stage%s_eval=%s afterok=%s script=scripts/orgslot/incremental2222/eval.sh\n' "$stage" "$evaluation" "$continuation" >&3
 scontrol show job "$train" "$continuation" "$evaluation"
 previous=$continuation
done
