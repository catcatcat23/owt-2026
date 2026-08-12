#!/bin/bash
#SBATCH --job-name=os-res-smoke
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/orgslot/logs/%x-%j.out
#SBATCH --error=slurm/orgslot/logs/%x-%j.err

set -euo pipefail
WORKDIR=${WORKDIR:-/gpfs/work/aac/sifansong/worktrees/orgslot_resolution}
DATA_ROOT=${DATA_ROOT:-/gpfs/work/aac/sifansong/data}
PYTHON=${PYTHON:-/gpfs/work/aac/sifansong/envs/abdpet/bin/python}
PROCESSED_DATA_ROOT=${PROCESSED_DATA_ROOT:-${DATA_ROOT}}
cd "${WORKDIR}"
source scripts/orgslot/configure_word_resolution_arm.sh
export WORKDIR DATA_ROOT PROCESSED_DATA_ROOT PYTHON
export LPIPS_STATE=${LPIPS_STATE:-${DATA_ROOT}/assets/owt_lpips_vgg16.pth}
export RUN_NAME=OrgSlot_${EXPERIMENT_TAG}_Loss3_Smoke_${SLURM_JOB_ID}
export OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
export N_GPU=2
export MICRO_BATCH=2
export ACCUM_ITER=1
export TARGET_EFFECTIVE_BATCH=4
export MAX_UPDATES=4
export WARMUP_UPDATES=0
export WORKERS=4
export SAVE_FREQ=1
export MAX_STEPS_PER_EPOCH=4
export ORGAN_ROI_PROBABILITY=1.0
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))

nvidia-smi -L
bash scripts/orgslot/run_word_resolution_loss3.sh
"${PYTHON}" tools/validate_orgslot_loss3_smoke.py --run-dir "${OUTPUT_DIR}"
