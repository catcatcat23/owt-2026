#!/bin/bash
#SBATCH --job-name=psemv3_triplet_word
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/psem_v3_triplet/slurm/logs/PSEM_v3/WORD_2D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/psem_v3_triplet/slurm/logs/PSEM_v3/WORD_2D/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/psem_v3_triplet
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Training_2D_224.csv
POSITIVE_COUNTS="2926 3247 3381 1088 2575 2505 4885 3704"
RUN_NAME=PSEMv3_TripletContext_Loss3_Delta01_Common8_WORD_2D_C8_T20_GPU2_B8A4_E1200
OUT_DIR=${WORKDIR}/Results/PSEM_v3_Triplet_Loss3/WORD_2D/${RUN_NAME}

mkdir -p ${OUT_DIR} ${WORKDIR}/slurm/logs/PSEM_v3/WORD_2D
cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PORT_NUM=${MASTER_PORT:-$((10000 + RANDOM % 50000))}

python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=${PORT_NUM} --use_env \
  main_pretrain_psem_triplet_loss3.py \
  --batch_size 8 --accum_iter 4 --model mae_vit_base_patch16-LA --mask_ratio 1.0 \
  --epochs 1200 --warmup_epochs 60 --blr 1e-4 --weight_decay 0.05 \
  --input_size 224 --num_classes 8 --arch_version v11 \
  --training_version v01 --token_factor 20 --loss_version L2-LPIPS \
  --positive_roi_loss_weight 0.25 --delta_loss_weight 0.1 \
  --roi_positive_sample_counts ${POSITIVE_COUNTS} --roi_frequency_alpha 0.5 \
  --roi_max_weight_ratio 4.0 --lpips_loss_weight 1.0 \
  --intensity_norm fixed_255 --data_path ${TRAIN_CSV} \
  --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 12 --save_freq 100 > ${OUT_DIR}/Result.txt 2>&1
