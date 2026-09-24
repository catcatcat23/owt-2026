#!/bin/bash
#SBATCH --job-name=owt_word_c8_eval
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Eval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Eval/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
RUN_DIR=${WORKDIR}/Results/Common8_Full/WORD_2D/Common8Full_HU-175_250_Token_mae_vit_base_patch16-LA_1e-4_8_224_20_v11_v01_L2-LPIPS_GPU2_32_1200
OUTPUT_DIR=${WORKDIR}/Results/Common8_Eval/WORD_2D/OWT-Joint-C8-Recon_ckpt1199

mkdir -p ${WORKDIR}/slurm/logs/Common8_Eval ${OUTPUT_DIR}
cd ${WORKDIR}

export PATH=${CONDA_ENV}/bin:${PATH}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python tools/evaluate_owt_common8.py \
  --dataset WORD \
  --test-csv /gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Test_2D_224.csv \
  --checkpoint ${RUN_DIR}/checkpoint-1199.pth \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 48 \
  --recon-threshold 0.02 \
  --seg-threshold 0.15 \
  --min-component-size 100 \
  --spacing 3.0 2.2857142857 2.2857142857 \
  --nsd-tolerance-mm 3.0
