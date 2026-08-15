#!/bin/bash
#SBATCH --job-name=xec_os448_jsall
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
#SBATCH --output=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD448_JointSeg_Train/%x_%j.out
#SBATCH --error=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD448_JointSeg_Train/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/antengcai23
WORKDIR=${ROOT}/worktrees/orgslotbank_slotseg
SEG_LOSS_TYPE=${SEG_LOSS_TYPE:-dice_bce}
FOCAL_ALPHA=${FOCAL_ALPHA:-0.75}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
LAMBDA_SEG=${LAMBDA_SEG:-0.01}
RUN_TAG=${RUN_TAG:-OrgSlot_WORD448_ROI20_JointSegAll_L0.01}
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/data
export PROCESSED_ROOT=${ROOT}/data/OWT_Common8_112_NATIVE/WORD
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_xec.json
export LPIPS_STATE=${ROOT}/worktrees/orgslotbank/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG LAMBDA_BG_SEG=0.25 SEG_SUPERVISION=all
export SEG_LOSS_TYPE FOCAL_ALPHA FOCAL_GAMMA
export ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export RUN_NAME OUTPUT_DIR
export N_GPU=2 MICRO_BATCH=8 ACCUM_ITER=12 TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 WEIGHT_DECAY=0.05
export WORKERS=10 SAVE_FREQ=100
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

nvidia-smi -L
${PYTHON} -c "import os, torch; count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); assert torch.cuda.is_available() and count == 2"
bash scripts/orgslot/run_word_112_448_a100.sh roi20

test -r "${OUTPUT_DIR}/checkpoint-802.pth"
${PYTHON} -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-802.pth', map_location='cpu'); a=c['args']; assert int(c['epoch']) == 802; assert int(getattr(a, 'max_optimizer_updates')) == 118800; assert getattr(a, 'seg_supervision') == 'all'; assert getattr(a, 'seg_loss_type') == '${SEG_LOSS_TYPE}'; assert float(getattr(a, 'lambda_seg')) == float('${LAMBDA_SEG}'); assert bool(getattr(a, 'organ_roi_aug')); print('formal checkpoint validated')"
