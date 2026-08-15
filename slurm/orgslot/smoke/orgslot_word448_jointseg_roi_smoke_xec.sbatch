#!/bin/bash
#SBATCH --job-name=xec_os448_js_smk
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD448_JointSeg_Smoke/%x_%j.out
#SBATCH --error=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD448_JointSeg_Smoke/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/antengcai23
WORKDIR=${ROOT}/worktrees/orgslotbank_slotseg
RUN_NAME=OrgSlot_WORD448_ROI100_JointSegAll_L0.01_Smoke_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/data
export PROCESSED_ROOT=${ROOT}/data/OWT_Common8_112_NATIVE/WORD
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_xec.json
export LPIPS_STATE=${ROOT}/worktrees/orgslotbank/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0.25 SEG_SUPERVISION=all
export ORGAN_ROI_PROBABILITY=1.0 TGR_MODE=legacy_batch
export RUN_NAME OUTPUT_DIR
export N_GPU=2 MICRO_BATCH=2 ACCUM_ITER=1 TARGET_EFFECTIVE_BATCH=4
export MAX_UPDATES=4 WARMUP_UPDATES=0 WORKERS=4 SAVE_FREQ=1
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

nvidia-smi -L
${PYTHON} -c "import os, torch; count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); assert torch.cuda.is_available() and count == 2"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
${PYTHON} tools/validate_orgslot_jointseg_smoke.py --run-dir "${OUTPUT_DIR}"
