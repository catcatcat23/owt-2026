#!/bin/bash
#SBATCH --job-name=owt_word_c8_3d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Full/WORD_3D_Fixfr4/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/Common8_Full/WORD_3D_Fixfr4/%x_%j.err

set -euo pipefail
WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
RESULT_PATH=${RESULT_PATH_OVERRIDE:-${WORKDIR}/Results/Common8_Full}
cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

PORT_NUM=${MASTER_PORT:-$((10000 + RANDOM % 50000))}
N_GPU=${SLURM_GPUS_ON_NODE:-2}
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_FAC=20
CLASSES=8
MODEL_NAME=mae_vit_basefix16_patch16-LA
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCH=${EPOCH:-1200}
WARMUP=${WARMUP:-60}
ARCH_VER=v11
TRAIN_VER=v01-3D-Fixfr4-TS1
LOSS_VER=L2-LPIPS
RUN_TAG=${RUN_TAG:-Common8Full}
DEFAULT_TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8/WORD/csv/WORD_Training_Fixfr4_224.csv
TRAIN_CSV=${TRAIN_CSV_OVERRIDE:-${DEFAULT_TRAIN_CSV}}

CUR_PATH=WORD_3D_Fixfr4
PROCESS_NAME=${RUN_TAG}_HU-175_250_Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_FAC}_${ARCH_VER}_${TRAIN_VER}_${LOSS_VER}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
OUT_DIR=${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}
mkdir -p ${OUT_DIR}/vis
echo "WORD Common8 Fixfr4 training started at $(date)"
echo "Train CSV: ${TRAIN_CSV}"
echo "Output dir: ${OUT_DIR}"

python -m torch.distributed.launch \
  --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env \
  main_pretrain.py \
  --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 \
  --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} \
  --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} \
  --arch_version ${ARCH_VER} --training_version ${TRAIN_VER} \
  --token_factor ${TOKEN_FAC} --loss_version ${LOSS_VER} \
  --intensity_norm fixed_255 --data_path ${TRAIN_CSV} \
  --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 12 --save_freq 100 \
  > ${OUT_DIR}/Result.txt 2>&1
echo "WORD Common8 Fixfr4 training finished at $(date)"
