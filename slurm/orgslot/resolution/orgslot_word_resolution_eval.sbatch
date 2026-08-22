#!/bin/bash
#SBATCH --job-name=os-res-eval
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/orgslot/logs/%x-%j.out
#SBATCH --error=slurm/orgslot/logs/%x-%j.err

set -euo pipefail
TRAIN_JOB_ID=${TRAIN_JOB_ID:?Set TRAIN_JOB_ID}
WORKDIR=${WORKDIR:-/gpfs/work/aac/sifansong/worktrees/orgslot_resolution}
DATA_ROOT=${DATA_ROOT:-/gpfs/work/aac/sifansong/data}
PYTHON=${PYTHON:-/gpfs/work/aac/sifansong/envs/abdpet/bin/python}
PROCESSED_DATA_ROOT=${PROCESSED_DATA_ROOT:-${DATA_ROOT}}
cd "${WORKDIR}"
source scripts/orgslot/configure_word_resolution_arm.sh
TRAIN_RUN=OrgSlot_${EXPERIMENT_TAG}_Loss3_ROI20_${TRAIN_JOB_ID}
TRAIN_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${TRAIN_RUN}
export PROCESSED_DATA_ROOT
CHECKPOINT=$(find "${TRAIN_DIR}" -maxdepth 1 -type f -name 'checkpoint-*.pth' | sort -V | tail -n 1)
if [[ -z "${CHECKPOINT}" ]]; then
  echo "No checkpoint found in ${TRAIN_DIR}" >&2
  exit 2
fi
export PYTHON CHECKPOINT
export TEST_CSV=${VAL_CSV}
export OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${TRAIN_RUN}_eval
export PREPROCESS_SUMMARY INPUT_SIZE GLOBAL_CROP_SIZE EXPECTED_SPACING
export BATCH_SIZE=4
export WORKERS=8

nvidia-smi -L
bash scripts/orgslot/run_word_resolution_eval.sh
