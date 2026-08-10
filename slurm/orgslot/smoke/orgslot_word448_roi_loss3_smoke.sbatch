#!/bin/bash
#SBATCH --job-name=os448r-l3-smk
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:2
#SBATCH --exclude=gpua800n2,gpua800n6
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/orgslot/logs/%x-%j.out
#SBATCH --error=slurm/orgslot/logs/%x-%j.err

set -euo pipefail
WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT_orgslot
RUN_NAME=OrgSlot_WORD448_ROI100_Loss3_Smoke_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd ${WORKDIR}
export PYTHON=/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python
export DATA_ROOT=/gpfs/work/aac/bolinren19/2026-07/DATA
export LPIPS_STATE=${WORKDIR}/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt
export LAMBDA_SEG=0.0
export POSITIVE_ROI_LOSS_WEIGHT=0.25
export ROI_POSITIVE_SAMPLE_COUNTS='4392 4875 5070 1631 3827 3753 7327 5556'
export ROI_FREQUENCY_DATASET_SIZE=28586
export ROI_FREQUENCY_ALPHA=0.5
export ROI_MAX_WEIGHT_RATIO=4.0
export ORGAN_ROI_PROBABILITY=1.0
export RUN_NAME OUTPUT_DIR
export N_GPU=2
export GPU_IDS=0,1
export MICRO_BATCH=2
export ACCUM_ITER=1
export TARGET_EFFECTIVE_BATCH=4
export MAX_UPDATES=4
export WARMUP_UPDATES=0
export WORKERS=4
export SAVE_FREQ=1
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))

nvidia-smi -L
${PYTHON} -c "import os, torch; expected=2; count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch_cuda=', torch.__version__, torch.version.cuda, 'count=', count); assert torch.cuda.is_available() and count == expected, 'required CUDA devices are not visible'; x=torch.ones(1, device='cuda'); assert x.item() == 1"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
${PYTHON} tools/validate_orgslot_loss3_smoke.py --run-dir ${OUTPUT_DIR}
