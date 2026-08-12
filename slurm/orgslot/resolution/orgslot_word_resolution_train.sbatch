#!/bin/bash
#SBATCH --job-name=os-res-train
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
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
export RUN_NAME=OrgSlot_${EXPERIMENT_TAG}_Loss3_ROI20_${SLURM_JOB_ID}
export OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
export N_GPU=2
export MICRO_BATCH=8
export ACCUM_ITER=12
export TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800
export WARMUP_UPDATES=5940
export WORKERS=10
export SAVE_FREQ=100
export ORGAN_ROI_PROBABILITY=0.2
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 20000))

nvidia-smi -L
bash scripts/orgslot/run_word_resolution_loss3.sh
