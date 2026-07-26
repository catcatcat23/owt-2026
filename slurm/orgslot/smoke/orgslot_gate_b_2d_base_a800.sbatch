#!/bin/bash
#SBATCH --job-name=orgslot-b2d
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/orgslot/logs/%x-%j.out

set -euo pipefail
cd /gpfs/work/aac/bolinren19/OD_OWT_orgslot
export OMP_NUM_THREADS=8

RESULT_DIR="Results/OrganSlotBank/_smoke/gate_b_2d_base_${SLURM_JOB_ID}"
MANIFEST_DIR="${RESULT_DIR}/manifests"
PYTHON="/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python"

nvidia-smi
"${PYTHON}" tools/split_orgslot_manifest.py \
  --source /gpfs/work/aac/bolinren19/2026-07/Training_A100_Final112.csv \
  --output-dir "${MANIFEST_DIR}" \
  --validation-fraction 0.1 \
  --seed 0

"${PYTHON}" main_pretrain_orgslot.py \
  --stage base \
  --data-path "${MANIFEST_DIR}/train.csv" \
  --val-data-path "${MANIFEST_DIR}/validation.csv" \
  --output-dir "${RESULT_DIR}" \
  --dimension 2D \
  --model-size base \
  --slot-tg-depth 1 \
  --loss-version L2 \
  --epochs 1 \
  --batch-size 1 \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --max-steps 1 \
  --workers 2 \
  --lr 1e-4 \
  --lambda-lpips 0 \
  --device cuda
