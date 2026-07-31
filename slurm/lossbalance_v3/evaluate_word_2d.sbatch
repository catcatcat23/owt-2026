#!/bin/bash
#SBATCH --job-name=eval_lbv3a_word_2d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3/slurm/logs/LossBalance_v3/WORD_2D_Eval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3/slurm/logs/LossBalance_v3/WORD_2D_Eval/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3
BASELINE_WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
RUN_NAME=LossBalanceV3a_Pos0.25_A0.5_R4.0_Token_mae_vit_base_patch16-LA_1e-4_C8_224_T20_v11_v01_L2-LPIPS_GPU2_B32_E1200
RUN_DIR=${WORKDIR}/Results/LossBalance_v3/WORD_2D/${RUN_NAME}
OUTPUT_DIR=${WORKDIR}/Results/LossBalance_v3/Eval/WORD_2D/ckpt1199

mkdir -p ${OUTPUT_DIR}
cd ${BASELINE_WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

test -f ${RUN_DIR}/checkpoint-1199.pth

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
