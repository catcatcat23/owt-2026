#!/bin/bash
#SBATCH --job-name=eval_abdpet_c4_2d
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/AbdAutoPet_Eval/2D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/2026-07/OD_OWT/slurm/logs/AbdAutoPet_Eval/2D/%x_%j.err

set -euo pipefail

WORKDIR=/gpfs/work/aac/bolinren19/2026-07/OD_OWT
CONDA_ENV=/gpfs/work/aac/bolinren19/.conda/envs/abdpet
CHECKPOINT=${CHECKPOINT_OVERRIDE:-${WORKDIR}/Results/AbdAutoPet_2D/Token_mae_vit_base_patch16-LA_1e-4_4_224_20_v11_v01_L2-LPIPS_GPU2_96_1200/checkpoint-1199.pth}
OUT_DIR=${OUT_DIR_OVERRIDE:-${WORKDIR}/Results/AbdAutoPet_Eval/2D/OWT-Joint-C4-Recon_ckpt1199}
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
