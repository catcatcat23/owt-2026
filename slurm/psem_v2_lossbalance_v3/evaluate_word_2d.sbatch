#!/bin/bash
#SBATCH --job-name=eval_psemv2_lbv3
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2_lossbalance_v3/slurm/logs/PSEM_v2/WORD_2D_Eval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2_lossbalance_v3/slurm/logs/PSEM_v2/WORD_2D_Eval/%x_%j.err

set -euo pipefail

PSEM_WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2_lossbalance_v3
BASELINE_WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
RUN_DIR=${PSEM_WORKDIR}/Results/PSEM_v2_LossBalance_v3/WORD_2D/PSEMv2_LossBalanceV3a_Query20_Common8_HU-175_250_Token_mae_vit_base_patch16-LA_1e-4_C8_224_T20_v11_v01_L2-LPIPS_GPU2_B32_E1200
OUTPUT_DIR=${PSEM_WORKDIR}/Results/PSEM_v2_LossBalance_v3/Eval/WORD_2D/ckpt1199

mkdir -p ${OUTPUT_DIR}
cd ${BASELINE_WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

test -f ${RUN_DIR}/checkpoint-1199.pth

# Each reconstruction query keeps the same token groups for every sample in a
# batch. PSEM therefore has zero token padding and is exactly equivalent to the
# original OWT forward path used by this common evaluator.
python tools/evaluate_owt_common8.py \
  --dataset WORD \
  --test-csv /gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Test_2D_224.csv \
  --checkpoint ${RUN_DIR}/checkpoint-1199.pth \
  --output-dir ${OUTPUT_DIR} \
  --model mae_vit_base_patch16-LA \
  --num-classes 8 \
  --token-factor 20 \
  --arch-version v11 \
  --training-version v01 \
  --input-size 224 \
  --batch-size 48 \
  --intensity-norm fixed_255 \
  --recon-threshold 0.02 \
  --seg-threshold 0.15 \
  --min-component-size 100 \
  --spacing 3.0 2.2857142857 2.2857142857 \
  --nsd-tolerance-mm 3.0 \
  > ${OUTPUT_DIR}/Result.txt 2>&1
