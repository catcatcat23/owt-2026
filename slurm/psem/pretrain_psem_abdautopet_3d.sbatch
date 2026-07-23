#!/bin/bash
#SBATCH --job-name=psem_abdpet_3d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/slurm/logs/PSEM_v1/AbdAutoPET_3D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/slurm/logs/PSEM_v1/AbdAutoPET_3D/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/Training_Fixfr4_A100_Final112.csv
RESULT_ROOT=${WORKDIR}/Results/PSEM_v1/AbdAutoPet_3D_Fixfr4
RUN_NAME=PSEM_v1_Token_mae_vit_basefix16_patch16-LA_1e-4_C4_224_T20_v11_v01-Fixfr4-TS1_L2-LPIPS_GPU2_B64_E1200
OUT_DIR=${RESULT_ROOT}/${RUN_NAME}

mkdir -p ${OUT_DIR}/vis
cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=$((10000 + RANDOM % 50000)) --use_env \
  main_pretrain_psem.py \
  --batch_size 64 --model mae_vit_basefix16_patch16-LA --mask_ratio 1.0 \
  --epochs 1200 --warmup_epochs 60 --blr 1e-4 --weight_decay 0.05 \
  --input_size 224 --num_classes 4 --arch_version v11 \
  --training_version v01-3D-Fixfr4-TS1 --token_factor 20 \
  --loss_version L2-LPIPS --intensity_norm per_sample \
  --data_path ${TRAIN_CSV} --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 8 --save_freq 100 > ${OUT_DIR}/Result.txt 2>&1
