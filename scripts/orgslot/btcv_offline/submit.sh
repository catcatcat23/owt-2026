#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPECTED_COMMIT:?}" "${RUN_ROOT:?}" "${BTCV_ROOT:?}" "${MAE_CHECKPOINT:?}"
[[ "$(id -un)" == sifansong ]]
cd "$EXPERIMENT_WORKDIR"
test "$(cat SOURCE_COMMIT)" = "$EXPECTED_COMMIT"
/usr/bin/python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["body_clipped_cases"] == [], "BTCV body crop coverage audit failed"' "$RUN_ROOT/body_center_audit.json"
test -s "$RUN_ROOT/body_center_audit.json"
test -s "$BTCV_ROOT/metadata/small_organ_roi_index.json"
mkdir -p "$RUN_ROOT"
set -o noclobber
exec 3> "$RUN_ROOT/jobs.txt"
printf 'commit=%s runtime=%s user=sifansong account=sifansong cluster=XEC qos=8gpus\n' "$EXPECTED_COMMIT" "$EXPERIMENT_WORKDIR" >&3
common=(--parsable --account=sifansong --qos=8gpus --nodes=1 --ntasks=1 --chdir="$EXPERIMENT_WORKDIR" --export=ALL --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err")
resources=(--partition=gpua800,gpua8001t --gres=gpu:a800:2 --cpus-per-task=16 --mem=128G --time=5-00:00:00)
train=$(sbatch "${common[@]}" "${resources[@]}" --job-name=BTCV512_SAM_softaux scripts/orgslot/btcv_offline/train.sh initial)
printf 'train=%s script=scripts/orgslot/btcv_offline/train.sh\n' "$train" >&3
resume=$(sbatch "${common[@]}" "${resources[@]}" --dependency="afterany:$train" --job-name=BTCV512_SAM_resume scripts/orgslot/btcv_offline/train.sh resume)
printf 'resume=%s afterany=%s\n' "$resume" "$train" >&3
evaljob=$(sbatch "${common[@]}" --partition=gpu4090,gpu40901t --gres=gpu:4090:1 --cpus-per-task=8 --mem=96G --time=1-00:00:00 --dependency="afterok:$resume" --job-name=BTCV512_SAM_eval scripts/orgslot/btcv_offline/eval.sh)
printf 'eval=%s afterok=%s script=scripts/orgslot/btcv_offline/eval.sh\n' "$evaljob" "$resume" >&3
for job in "$train" "$resume" "$evaljob"; do scontrol show job "$job"; done
