#!/bin/bash
#SBATCH --job-name=sf_eval_pv3_nan
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/PSEM_v3/AbdAutoPET_2D_Nanfix_Eval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/PSEM_v3/AbdAutoPET_2D_Nanfix_Eval/%x_%j.err

set -euo pipefail

ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/psem_v3_triplet_nanfix
CONDA_ENV=${ROOT}/envs/abdpet
TEST_CSV=${ROOT}/csv/AbdAutoPET/Test_A100_Final112_xec.csv
TMPDIR=${ROOT}/artifacts/tmp/${SLURM_JOB_ID}
RUN_NAME=PSEMv3_TripletContext_Loss3_Delta01_AbdAutoPET_2D_C4_T20_GPU4_B12A4_E1200_RESUME_CKPT400_FP32FIX
RUN_DIR=${ROOT}/Results/PSEM_v3_Triplet_Loss3/AbdAutoPET_2D/${RUN_NAME}
OUTPUT_DIR=${ROOT}/Results/PSEM_v3_Triplet_Loss3/Eval/AbdAutoPET_2D/ckpt1199_resume400_fp32fix

mkdir -p "${OUTPUT_DIR}" "${TMPDIR}"
cd "${WORKDIR}"
git rev-parse HEAD
git status --short
sha256sum VQ/lpips.py engine_pretrain_psem_triplet_loss3.py
export PATH=${CONDA_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
export TMPDIR

test -x "${CONDA_ENV}/bin/python"
test -r "${TEST_CSV}"
test -f "${RUN_DIR}/checkpoint-1199.pth"
test -f "${WORKDIR}/tools/evaluate_owt_common8.py"
test ! -e "${OUTPUT_DIR}/step3_segmentation_summary.csv"
python -c "import os, torch; expected=1; count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch_cuda=', torch.__version__, torch.version.cuda, 'count=', count); assert torch.cuda.is_available() and count == expected; x=torch.ones(1, device='cuda'); assert x.item() == 1"
python -c "import torch; c=torch.load('${RUN_DIR}/checkpoint-1199.pth', map_location='cpu'); assert int(c['epoch']) == 1199; print('checkpoint epoch validated')"

python tools/evaluate_owt_common8.py \
  --dataset AbdAutoPet --test-csv "${TEST_CSV}" \
  --checkpoint "${RUN_DIR}/checkpoint-1199.pth" --output-dir "${OUTPUT_DIR}" \
  --model mae_vit_base_patch16-LA --num-classes 4 \
  --class-names label_1 label_2 label_3 label_4 \
  --intensity-norm per_sample --batch-size 96 \
  --seg-threshold 0.15 --min-component-size 20 \
  > "${OUTPUT_DIR}/Result.txt" 2>&1
