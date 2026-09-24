#!/bin/bash
#SBATCH --job-name=eval_word_c8_3d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Eval/WORD_3D_Fixfr4/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Eval/WORD_3D_Fixfr4/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
DEFAULT_CHECKPOINT=${WORKDIR}/Results/Common8_Full/WORD_3D_Fixfr4/Common8Full_HU-175_250_Token_mae_vit_basefix16_patch16-LA_1e-4_8_224_20_v11_v01-3D-Fixfr4-TS1_L2-LPIPS_GPU2_32_1200/checkpoint-1199.pth
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
OUT_DIR=${OUT_DIR:-${WORKDIR}/Results/Common8_Eval/WORD_3D_Fixfr4/OWT-Joint-C8-Recon3D_ckpt1199}

cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
mkdir -p ${OUT_DIR}
test -f ${CHECKPOINT}

python tools/evaluate_owt_common8_3d.py \
  --dataset WORD \
  --test-csv /gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Test_2D_224.csv \
  --checkpoint ${CHECKPOINT} \
  --output-dir ${OUT_DIR} \
  --batch-size 8 \
  > ${OUT_DIR}/Result.txt 2>&1
