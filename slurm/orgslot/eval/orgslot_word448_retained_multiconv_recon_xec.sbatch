#!/bin/bash
#SBATCH --job-name=xec_eval_os448_rmcf_rec
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD448_RetainedMultiFocal/%x_%j.out
#SBATCH --error=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD448_RetainedMultiFocal/%x_%j.err

set -euo pipefail
: "${TRAIN_JOB_ID:?submit with TRAIN_JOB_ID}"
ROOT=/gpfs/work/aac/antengcai23
WORKDIR=${ROOT}/worktrees/orgslotbank_slotseg
ENV=${ROOT}/envs/abdpet
RUN_TAG=OrgSlot_WORD448_ROI20_RetainedMultiConv_Focal_L0.1
RUN=${RUN_TAG}_${TRAIN_JOB_ID}
CHECKPOINT=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN}/checkpoint-802.pth
OUTPUT=${WORKDIR}/Results/OrganSlotBank/evaluation/Common8/WORD_2D/${RUN}_reconstruction_fixedthr002
cd "${WORKDIR}"
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
python -m tools.eval_common8_orgslot_reconstruction_threshold   --method orgslot --checkpoint "${CHECKPOINT}"   --test-csv ${ROOT}/data/OWT_Common8_112_NATIVE/WORD/csv/WORD_Test_2D_native112.csv   --output-dir "${OUTPUT}" --class-map configs/orgslot/common8_offline.json   --fusion-mode linear_sqrt --input-size 448 --global-crop-size 448   --batch-size 4 --workers 8 --threshold 0.02 --min-size 20   --opening-radius 1 --class-ids 1,2,3,4,5,6,7,8 --device cuda
