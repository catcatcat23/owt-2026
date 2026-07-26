#!/bin/bash
#SBATCH --job-name=orgslot-gate-a
#SBATCH --partition=cpudebug
#SBATCH --account=sifansong
#SBATCH --qos=cpudebug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/orgslot/logs/%x-%j.out

set -euo pipefail
cd /gpfs/work/aac/bolinren19/OD_OWT_orgslot
export OMP_NUM_THREADS=4
/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python main_pretrain_orgslot.py \
  --synthetic \
  --stage base \
  --epochs 2 \
  --output-dir Results/OrganSlotBank/_smoke/gate_a_cpu_${SLURM_JOB_ID} \
  --device cpu
