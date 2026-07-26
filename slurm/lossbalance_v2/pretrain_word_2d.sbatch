#!/bin/bash
#SBATCH --job-name=lbv2_word_2d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v2/slurm/logs/LossBalance_v2/WORD_2D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v2/slurm/logs/LossBalance_v2/WORD_2D/%x_%j.err

set -euo pipefail
WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT_lossbalance_v2
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
RESULT_PATH=${RESULT_PATH_OVERRIDE:-${WORKDIR}/Results/LossBalance_v2}
TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Training_2D_224.csv
cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

PORT_NUM=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
N_GPU=${SLURM_GPUS_ON_NODE:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCH=${EPOCH:-1200}
WARMUP=${WARMUP:-60}
SAVE_FREQ=${SAVE_FREQ:-100}
ROI_WEIGHT=${ROI_WEIGHT:-1.0}
FREQ_ALPHA=${FREQ_ALPHA:-0.5}
MAX_WEIGHT_RATIO=${MAX_WEIGHT_RATIO:-4.0}
POSITIVE_COUNTS=${POSITIVE_COUNTS:-"2926 3247 3381 1088 2575 2505 4885 3704"}
RUN_TAG=${RUN_TAG:-LossBalanceV2}
PROCESS_NAME=${RUN_TAG}_ROI${ROI_WEIGHT}_A${FREQ_ALPHA}_R${MAX_WEIGHT_RATIO}_Token_mae_vit_base_patch16-LA_1e-4_8_224_20_v11_v01_L2-LPIPS_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
OUT_DIR=${RESULT_PATH}/WORD_2D/${PROCESS_NAME}
mkdir -p ${OUT_DIR}/vis

python -m torch.distributed.launch \
  --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env \
  main_pretrain_lossbalance_v2.py \
  --batch_size ${BATCH_SIZE} --model mae_vit_base_patch16-LA \
  --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} \
  --blr 1e-4 --weight_decay 0.05 --input_size 224 --num_classes 8 \
  --arch_version v11 --training_version v01 --token_factor 20 \
  --loss_version L2-LPIPS --intensity_norm fixed_255 \
  --roi_loss_weight ${ROI_WEIGHT} --roi_positive_sample_counts ${POSITIVE_COUNTS} \
  --roi_frequency_alpha ${FREQ_ALPHA} --roi_max_weight_ratio ${MAX_WEIGHT_RATIO} \
  --lpips_loss_weight 1.0 --data_path ${TRAIN_CSV} \
  --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 12 --save_freq ${SAVE_FREQ} \
  > ${OUT_DIR}/Result.txt 2>&1

echo "WORD 2D LossBalance training finished at $(date)"
