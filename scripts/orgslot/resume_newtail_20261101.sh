#!/usr/bin/env bash
# Continue the existing ea8780a experiment; do not migrate it to newer model code.
set -euo pipefail
export EXPERIMENT_WORKDIR=/gpfs/work/aac/sifansong/worktrees/inc44_newtail_ea8780a
export EXPECTED_COMMIT=ea8780adb46e2f6aa682dd0169a1277c4e3b4406
export ARM_F_TARGET=sifan-xec INCREMENTAL_ARCHITECTURE=sam_soft_aux
export NEW_DATA_ROOT=/gpfs/work/aac/sifansong/artifacts/inc44_train_data_20261006
export RUN_ROOT=/gpfs/work/aac/sifansong/artifacts/inc44_newtail_20261027
export STAGE1_RUN=/gpfs/work/aac/sifansong/artifacts/inc44_full_20261006/SAM/stage1_156536
export STAGE1_TEST=/gpfs/work/aac/sifansong/artifacts/inc44_full_20261006/SAM/stage1_evaluation/test
export BACKGROUND_POLICY=separation STAGE2_SHARED_SEGMENTATION=slot_private
export STAGE2_PIXEL_DECODER=new4 STAGE2_READOUT_ADAPTER=none STAGE2_GPU_PREFLIGHT=0
export INCREMENTAL_NPROC_PER_NODE=2 ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=6
export EVALUATE_RECONSTRUCTION=1 STAGE2_TAIL=new4
export RESUME_CHECKPOINT=$RUN_ROOT/stage2_separation_174375/checkpoint-760.pth
[[ $(id -un) == sifansong ]]
test "$(cat "$EXPERIMENT_WORKDIR/SOURCE_COMMIT")" = "$EXPECTED_COMMIT"
python3 - "$EXPERIMENT_WORKDIR" "$RUN_ROOT/resume_source_hashes.json" <<'PY'
import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1])
for name,sha in json.load(open(sys.argv[2])).items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
print('Pinned runtime hashes verified')
PY
set -o noclobber
exec 3> "$RUN_ROOT/resume_20261101_jobs.txt"
printf 'commit=%s runtime=%s account=sifansong cluster=XEC qos=8gpus resume=%s\n' "$EXPECTED_COMMIT" "$EXPERIMENT_WORKDIR" "$RESUME_CHECKPOINT" >&3
common=(--parsable --account=sifansong --qos=8gpus --partition=gpu40901t,gpu4090 --chdir="$EXPERIMENT_WORKDIR" --export=ALL --output="$RUN_ROOT/%j.out" --error="$RUN_ROOT/%j.err")
train=$(sbatch "${common[@]}" --gres=gpu:4090:2 --cpus-per-task=12 --mem=192G --time=5-00:00:00 --job-name=SAM_newtail_resume "$EXPERIMENT_WORKDIR/slurm/orgslot/train/incremental44_stage2_sip.sbatch")
printf 'train=%s script=slurm/orgslot/train/incremental44_stage2_sip.sbatch\n' "$train" >&3
export STAGE2_RUN="$RUN_ROOT/stage2_separation_$train" OUTPUT_ROOT="$RUN_ROOT/evaluation"
evaljob=$(sbatch "${common[@]}" --gres=gpu:4090:1 --cpus-per-task=10 --mem=128G --time=1-00:00:00 --dependency="afterok:$train" --job-name=SAM_newtail_eval "$EXPERIMENT_WORKDIR/slurm/orgslot/eval/incremental44_stage2.sbatch")
printf 'eval=%s afterok=%s stage2_run=%s\n' "$evaljob" "$train" "$STAGE2_RUN" >&3
# Replace the stale dependency only after both replacement jobs exist.
scontrol show job "$train"
scontrol show job "$evaljob"
scancel 174376
