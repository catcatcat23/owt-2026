#!/bin/bash
#SBATCH --job-name=eval_word_3d_smoke
#SBATCH --partition=gpudebug
#SBATCH --account=sifansong
#SBATCH --qos=gpudebug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Eval/Smoke/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Eval/Smoke/%x_%j.err

set -euo pipefail
WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
CHECKPOINT=${WORKDIR}/Results/_smoke/Common8_3D/WORD_3D_Fixfr4/Smoke64_HU-175_250_Token_mae_vit_basefix16_patch16-LA_1e-4_8_224_20_v11_v01-3D-Fixfr4-TS1_L2-LPIPS_GPU1_4_1/checkpoint-0.pth
OUT_DIR=${WORKDIR}/Results/_smoke/Common8_3D_Eval/WORD

cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

python tools/evaluate_owt_common8_3d.py \
  --dataset WORD \
  --test-csv /gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Test_2D_224.csv \
  --checkpoint ${CHECKPOINT} \
  --output-dir ${OUT_DIR} \
  --batch-size 4 \
  --max-cases 1 \
  --skip-lpips
