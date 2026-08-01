#!/bin/bash
#SBATCH --job-name=eval_psemv2a_auto2d
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2/slurm/logs/PSEM_v2/AbdAutoPET_2D_Eval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT_psem_v2/slurm/logs/PSEM_v2/AbdAutoPET_2D_Eval/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
CHECKPOINT=${CHECKPOINT_OVERRIDE:-/gpfs/work/aac/bolinren19/OD_OWT_psem_v2/Results/PSEM_v2/AbdAutoPet_2D/PSEM_v2a_Query20_Token_mae_vit_base_patch16-LA_1e-4_C4_224_T20_v11_v01_L2-LPIPS_GPU2_B96_E1200/checkpoint-1199.pth}
OUT_DIR=${OUT_DIR_OVERRIDE:-/gpfs/work/aac/bolinren19/OD_OWT_psem_v2/Results/PSEM_v2/Eval/AbdAutoPet_2D/ckpt1199}
MAX_CASES=${MAX_CASES:-0}
CASE_START=${CASE_START:-0}
SKIP_LPIPS=${SKIP_LPIPS:-0}
BATCH_SIZE=${BATCH_SIZE:-96}

cd ${WORKDIR}
export PATH=${CONDA_ENV}/bin:${PATH}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
mkdir -p ${OUT_DIR}
test -f ${CHECKPOINT}

EXTRA_ARGS=()
if [[ ${MAX_CASES} -gt 0 ]]; then
  EXTRA_ARGS+=(--max-cases ${MAX_CASES})
fi
if [[ ${CASE_START} -gt 0 ]]; then
  EXTRA_ARGS+=(--case-start ${CASE_START})
fi
if [[ ${SKIP_LPIPS} -eq 1 ]]; then
  EXTRA_ARGS+=(--skip-lpips)
fi

python tools/evaluate_owt_common8.py \
  --dataset AbdAutoPet \
  --test-csv /gpfs/work/aac/bolinren19/2026-07/Test_A100_Final112.csv \
  --checkpoint ${CHECKPOINT} \
  --output-dir ${OUT_DIR} \
  --model mae_vit_base_patch16-LA \
  --num-classes 4 \
  --class-names label_1 label_2 label_3 label_4 \
  --intensity-norm per_sample \
  --batch-size ${BATCH_SIZE} \
  --seg-threshold 0.15 \
  --min-component-size 20 \
  "${EXTRA_ARGS[@]}" \
  > ${OUT_DIR}/Result.txt 2>&1
