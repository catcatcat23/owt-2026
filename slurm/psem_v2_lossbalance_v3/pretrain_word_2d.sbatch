#!/bin/bash
#SBATCH --job-name=psemv2_lbv3_word
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2_lossbalance_v3/slurm/logs/PSEM_v2/WORD_2D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2_lossbalance_v3/slurm/logs/PSEM_v2/WORD_2D/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2_lossbalance_v3
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Training_2D_224.csv
POSITIVE_COUNTS="2926 3247 3381 1088 2575 2505 4885 3704"
RESULT_ROOT=${WORKDIR}/Results/PSEM_v2_LossBalance_v3/WORD_2D
RUN_NAME=PSEMv2_LossBalanceV3a_Query20_Common8_HU-175_250_Token_mae_vit_base_patch16-LA_1e-4_C8_224_T20_v11_v01_L2-LPIPS_GPU2_B32_E1200
OUT_DIR=${RESULT_ROOT}/${RUN_NAME}

mkdir -p ${OUT_DIR}/vis
cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

PORT_NUM=${MASTER_PORT:-$((10000 + RANDOM % 50000))}

echo "PSEM-v2a WORD Common8 2D training started at $(date)"
echo "Train CSV: ${TRAIN_CSV}"
echo "Output dir: ${OUT_DIR}"

python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=${PORT_NUM} --use_env \
  main_pretrain_psem_lossbalance_v3.py \
  --batch_size 32 --model mae_vit_base_patch16-LA --mask_ratio 1.0 \
  --epochs 1200 --warmup_epochs 60 --blr 1e-4 --weight_decay 0.05 \
  --input_size 224 --num_classes 8 --arch_version v11 \
  --training_version v01 --token_factor 20 --loss_version L2-LPIPS \
  --positive_roi_loss_weight 0.25 --roi_positive_sample_counts ${POSITIVE_COUNTS} \
  --roi_frequency_alpha 0.5 --roi_max_weight_ratio 4.0 --lpips_loss_weight 1.0 \
  --intensity_norm fixed_255 --data_path ${TRAIN_CSV} \
  --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 12 --save_freq 100 > ${OUT_DIR}/Result.txt 2>&1

echo "PSEM-v2a WORD Common8 2D training finished at $(date)"
