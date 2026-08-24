#!/bin/bash
#SBATCH --job-name=sip_focal_c500_rec
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_head_ab/slurm/orgslot/logs/WORD448_FocalCkpt500/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_head_ab/slurm/orgslot/logs/WORD448_FocalCkpt500/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/bolinren19
WORKDIR=${ROOT}/OD_OWT/.worktrees/orgslot_head_ab
DATA=${ROOT}/2026-07/DATA/processed/OWT_Common8_112_NATIVE/WORD
RUN=OrgSlot_WORD448_ROI20_RetainedMultiConv_Focal_L0.1_119641_ckpt500
CHECKPOINT=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/OrgSlot_WORD448_ROI20_RetainedMultiConv_Focal_L0.1_119641/checkpoint-500.pth
OUTPUT=${WORKDIR}/Results/OrganSlotBank/evaluation/Common8/WORD_2D/${RUN}_reconstruction_fixedthr002
PYTHON=${ROOT}/.conda/envs/abdpet/bin/python
cd "${WORKDIR}"

export PATH=${ROOT}/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/.conda/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
test "$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)" = "aca81047809978f6a1f818c468202485895a48ab20f2750182e54124fdf02dab"
test ! -e "${OUTPUT}"
nvidia-smi -L
${PYTHON} -m tools.eval_common8_orgslot_reconstruction_threshold \
  --method orgslot \
  --checkpoint "${CHECKPOINT}" \
  --test-csv "${DATA}/csv/WORD_Test_2D_native112.csv" \
  --preprocess-summary "${DATA}/metadata/preprocess_summary.json" \
  --expected-spacing 1 1 2 \
  --output-dir "${OUTPUT}" \
  --class-map configs/orgslot/common8_offline.json \
  --fusion-mode linear_sqrt \
  --input-size 448 --global-crop-size 448 \
  --batch-size 4 --workers 8 \
  --threshold 0.02 --min-size 20 --opening-radius 1 \
  --class-ids 1,2,3,4,5,6,7,8 --device cuda
${PYTHON} tools/validate_word_orgslot_resolution_eval.py \
  --eval-dir "${OUTPUT}" --expected-input-size 448 --expected-global-crop 448 \
  --expected-spacing 1 1 2 --expected-cases 24
