#!/bin/bash
#SBATCH --job-name=eval_psemv3_auto
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/psem_v3_triplet/slurm/logs/PSEM_v3/AbdAutoPET_2D_Eval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/psem_v3_triplet/slurm/logs/PSEM_v3/AbdAutoPET_2D_Eval/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/psem_v3_triplet
BASELINE_WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
RUN_NAME=PSEMv3_TripletContext_Loss3_Delta01_AbdAutoPET_2D_C4_T20_GPU4_B12A4_E1200
RUN_DIR=${WORKDIR}/Results/PSEM_v3_Triplet_Loss3/AbdAutoPET_2D/${RUN_NAME}
OUTPUT_DIR=${WORKDIR}/Results/PSEM_v3_Triplet_Loss3/Eval/AbdAutoPET_2D/ckpt1199

mkdir -p ${OUTPUT_DIR} ${WORKDIR}/slurm/logs/PSEM_v3/AbdAutoPET_2D_Eval
cd ${BASELINE_WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

python -c "import os, torch; expected=1; count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch_cuda=', torch.__version__, torch.version.cuda, 'count=', count); assert torch.cuda.is_available() and count == expected, 'required CUDA device is not visible'; x=torch.ones(1, device='cuda'); assert x.item() == 1"
test -f ${RUN_DIR}/checkpoint-1199.pth
python tools/evaluate_owt_common8.py \
  --dataset AbdAutoPet \
  --test-csv /gpfs/work/aac/bolinren19/2026-07/Test_A100_Final112.csv \
  --checkpoint ${RUN_DIR}/checkpoint-1199.pth \
  --output-dir ${OUTPUT_DIR} \
  --model mae_vit_base_patch16-LA \
  --num-classes 4 \
  --class-names label_1 label_2 label_3 label_4 \
  --intensity-norm per_sample \
  --batch-size 96 \
  --seg-threshold 0.15 \
  --min-component-size 20 \
  > ${OUTPUT_DIR}/Result.txt 2>&1
