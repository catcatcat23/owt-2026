#!/bin/bash
#SBATCH --job-name=lbv3a_smoke2d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3/slurm/logs/LossBalance_v3/WORD_2D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3/slurm/logs/LossBalance_v3/WORD_2D/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v3
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Training_2D_224.csv
RESULT_ROOT=${WORKDIR}/Results/LossBalance_v3
POSITIVE_WEIGHT=0.25
FREQ_ALPHA=0.5
MAX_WEIGHT_RATIO=4.0
POSITIVE_COUNTS="2926 3247 3381 1088 2575 2505 4885 3704"
RUN_NAME=LossBalanceV3a_Pos${POSITIVE_WEIGHT}_A${FREQ_ALPHA}_R${MAX_WEIGHT_RATIO}_Smoke_Token_mae_vit_base_patch16-LA_1e-4_C8_224_T20_v11_v01_L2-LPIPS_GPU2_B32_E1
OUT_DIR=${RESULT_ROOT}/WORD_2D/${RUN_NAME}

mkdir -p ${OUT_DIR}/vis
cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PORT_NUM=$((10000 + RANDOM % 50000))

python -m torch.distributed.launch \
  --nproc_per_node=2 --master_port=${PORT_NUM} --use_env \
  main_pretrain_lossbalance_v3.py \
  --batch_size 32 --model mae_vit_base_patch16-LA \
  --mask_ratio 1.0 --epochs 1 --warmup_epochs 0 \
  --blr 1e-4 --weight_decay 0.05 --input_size 224 --num_classes 8 \
  --arch_version v11 --training_version v01 --token_factor 20 \
  --loss_version L2-LPIPS --intensity_norm fixed_255 \
  --positive_roi_loss_weight ${POSITIVE_WEIGHT} \
  --roi_positive_sample_counts ${POSITIVE_COUNTS} \
  --roi_frequency_alpha ${FREQ_ALPHA} \
  --roi_max_weight_ratio ${MAX_WEIGHT_RATIO} \
  --lpips_loss_weight 1.0 --data_path ${TRAIN_CSV} \
  --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 12 --save_freq 1 \
  > ${OUT_DIR}/Result.txt 2>&1

python tools/validate_lossbalance_v3_smoke.py --run-dir ${OUT_DIR}
