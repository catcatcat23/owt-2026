#!/bin/bash
#SBATCH --job-name=psem_abdpet_2d_smoke
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/slurm/logs/PSEM_v1/AbdAutoPET_2D_Smoke/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/slurm/logs/PSEM_v1/AbdAutoPET_2D_Smoke/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/Training_A100_Final112.csv
RESULT_ROOT=${WORKDIR}/Results/PSEM_v1/_smoke/AbdAutoPet_2D
RUN_NAME=PSEM_v1_AbdAutoPet_2D_C4_T20_B8_E2
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
  --batch_size 8 --model mae_vit_base_patch16-LA --mask_ratio 1.0 \
  --epochs 2 --warmup_epochs 1 --blr 1e-4 --weight_decay 0.05 \
  --input_size 224 --num_classes 4 --arch_version v11 \
  --training_version v01 --token_factor 20 --loss_version L2-LPIPS \
  --intensity_norm per_sample --max_train_samples 64 \
  --data_path ${TRAIN_CSV} --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 4 --save_freq 1 > ${OUT_DIR}/Result.txt 2>&1
