#!/bin/bash
#SBATCH --job-name=os448r20-l3
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm/orgslot/logs/%x-%j.out
#SBATCH --error=slurm/orgslot/logs/%x-%j.err

set -euo pipefail
WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT_orgslot
RUN_NAME=OrgSlot_WORD448_ROI20_Loss3_Pos0.25_A0.5_R4_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd ${WORKDIR}
export PYTHON=/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python
export DATA_ROOT=/gpfs/work/aac/bolinren19/2026-07/DATA
export LPIPS_STATE=${WORKDIR}/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt
export LAMBDA_SEG=1.0
export POSITIVE_ROI_LOSS_WEIGHT=0.25
export ROI_POSITIVE_SAMPLE_COUNTS='4392 4875 5070 1631 3827 3753 7327 5556'
export ROI_FREQUENCY_DATASET_SIZE=28586
export ROI_FREQUENCY_ALPHA=0.5
export ROI_MAX_WEIGHT_RATIO=4.0
export ORGAN_ROI_PROBABILITY=0.2
export RUN_NAME OUTPUT_DIR
export N_GPU=2
export GPU_IDS=0,1
export MICRO_BATCH=8
export ACCUM_ITER=12
export TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800
export WARMUP_UPDATES=5940
export WORKERS=10
export SAVE_FREQ=100
export MASTER_PORT=$((29000 + SLURM_JOB_ID % 20000))

nvidia-smi
bash scripts/orgslot/run_word_112_448_a100.sh roi20
