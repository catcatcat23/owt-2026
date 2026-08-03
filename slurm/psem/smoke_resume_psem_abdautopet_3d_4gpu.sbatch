#!/bin/bash
#SBATCH --job-name=psem3d_r4_smoke
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/slurm/logs/PSEM_v1/AbdAutoPET_3D_Smoke/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/slurm/logs/PSEM_v1/AbdAutoPET_3D_Smoke/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
TRAIN_CSV=/gpfs/work/aac/bolinren19/2026-07/Training_Fixfr4_A100_Final112.csv
SOURCE_DIR=${WORKDIR}/Results/PSEM_v1/_smoke/AbdAutoPet_3D_Fixfr4/PSEM_v1_AbdAutoPet_3D_C4_T20_B4_E2
RESULT_ROOT=${WORKDIR}/Results/PSEM_v1/_smoke/AbdAutoPet_3D_Fixfr4
RUN_NAME=PSEM_v1_AbdAutoPet_3D_Resume2GPUto4GPU_C4_T20_GB8
OUT_DIR=${RESULT_ROOT}/${RUN_NAME}
RESUME_CKPT=${SOURCE_DIR}/checkpoint-0.pth

if [[ ! -s ${RESUME_CKPT} ]]; then
  echo "Missing or empty resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

mkdir -p ${OUT_DIR}/vis
cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Resume a 2-GPU checkpoint on 4 GPUs while keeping global batch 8:
# 2 GPUs x 4 samples = 4 GPUs x 2 samples.
python -m torch.distributed.launch \
  --nproc_per_node=4 --master_port=$((10000 + RANDOM % 50000)) --use_env \
  main_pretrain_psem.py \
  --batch_size 2 --model mae_vit_basefix16_patch16-LA --mask_ratio 1.0 \
  --epochs 2 --warmup_epochs 1 --blr 1e-4 --weight_decay 0.05 \
  --input_size 224 --num_classes 4 --arch_version v11 \
  --training_version v01-3D-Fixfr4-TS1 --token_factor 20 \
  --loss_version L2-LPIPS --intensity_norm per_sample \
  --max_train_samples 32 --data_path ${TRAIN_CSV} \
  --output_dir ${OUT_DIR}/ --log_dir ${OUT_DIR}/ \
  --num_workers 2 --save_freq 1 --resume ${RESUME_CKPT} \
  > ${OUT_DIR}/Result.txt 2>&1
