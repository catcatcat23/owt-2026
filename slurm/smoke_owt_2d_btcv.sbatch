#!/bin/bash
#SBATCH --job-name=smoke_btcv_2d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/smoke/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/smoke/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
RESULT_PATH=${WORKDIR}/Results/_smoke

mkdir -p ${WORKDIR}/slurm/logs/smoke
cd ${WORKDIR}

export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

PORT_NUM=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
N_GPU=${SLURM_GPUS_ON_NODE:-2}
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20

CUR_PATH=BTCV_2D
CLASSES=13
MODEL_NAME=mae_vit_base_patch16-LA
BATCH_SIZE=16
EPOCH=2
WARMUP=1
ARCH_Ver=v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS
TRAIN_CSV_PATH=/gpfs/work/aac/bolinren19/2026-07/BTCV_Training_2D_224.csv

PROCESS_NAME=Smoke_Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
OUT_DIR=${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}
mkdir -p ${OUT_DIR}/vis

echo "Smoke BTCV 2D started at $(date)"
echo "Train CSV: ${TRAIN_CSV_PATH}"
echo "Output dir: ${OUT_DIR}"

python -m torch.distributed.launch \
  --nproc_per_node=${N_GPU} \
  --master_port=${PORT_NUM} \
  --use_env \
  main_pretrain.py \
  --batch_size ${BATCH_SIZE} \
  --model ${MODEL_NAME} \
  --mask_ratio 1.0 \
  --epochs ${EPOCH} \
  --warmup_epochs ${WARMUP} \
  --blr ${BASE_LR} \
  --weight_decay 0.05 \
  --input_size ${INPUT_SIZE} \
  --num_classes ${CLASSES} \
  --arch_version ${ARCH_Ver} \
  --training_version ${TRAIN_Ver} \
  --token_factor ${TOKEN_Fac} \
  --loss_version ${LOSS_Ver} \
  --data_path ${TRAIN_CSV_PATH} \
  --output_dir ${OUT_DIR}/ \
  --log_dir ${OUT_DIR}/ \
  --save_freq 1 \
  > ${OUT_DIR}/Result.txt 2>&1

echo "Smoke BTCV 2D finished at $(date)"
