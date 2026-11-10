#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${RUN_ROOT:?}" "${BTCV_ROOT:?}" "${MAE_CHECKPOINT:?}" "${RESENC_CHECKPOINT:?}" "${VARIANT:?}"
[[ "$(id -un)" == sifansong ]]
cd "$EXPERIMENT_WORKDIR"
test "$(cat SOURCE_COMMIT)" = "$EXPECTED_COMMIT"
/usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["seed"]==43; assert d["splits"]["Test"]==["img0008","img0029","img0034","img0036","img0038","img0040"]' "$BTCV_ROOT/split.json"
test -s "$BTCV_ROOT/metadata/small_organ_roi_index.json"
mkdir -p "$RUN_ROOT"
set -o noclobber
exec 3> "$RUN_ROOT/jobs.txt"
printf 'commit=%s runtime=%s user=sifansong account=sifansong cluster=XEC qos=8gpus\n' "$EXPECTED_COMMIT" "$EXPERIMENT_WORKDIR" >&3
common=(--parsable --account=sifansong --qos=8gpus --nodes=1 --ntasks=1 --chdir="$EXPERIMENT_WORKDIR" --export=ALL --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err")
resources=(--partition=gpua800,gpua8001t --gres=gpu:a800:4 --cpus-per-task=24 --mem=192G --time=5-00:00:00)
train=$(sbatch "${common[@]}" "${resources[@]}" --job-name=BTCV43_${VARIANT}_softaux scripts/orgslot/btcv_seed43/train.sh initial)
printf 'train=%s script=scripts/orgslot/btcv_seed43/train.sh\n' "$train" >&3
resume=$(sbatch "${common[@]}" "${resources[@]}" --dependency="afterany:$train" --job-name=BTCV43_${VARIANT}_resume scripts/orgslot/btcv_seed43/train.sh resume)
printf 'resume=%s afterany=%s\n' "$resume" "$train" >&3
evaljob=$(sbatch "${common[@]}" --partition=gpu4090,gpu40901t --gres=gpu:4090:1 --cpus-per-task=8 --mem=96G --time=1-00:00:00 --dependency="afterok:$resume" --job-name=BTCV43_${VARIANT}_eval scripts/orgslot/btcv_offline/eval.sh)
printf 'eval=%s afterok=%s script=scripts/orgslot/btcv_offline/eval.sh\n' "$evaljob" "$resume" >&3
for job in "$train" "$resume" "$evaljob"; do scontrol show job "$job"; done
