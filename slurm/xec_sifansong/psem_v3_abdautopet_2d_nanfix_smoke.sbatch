#!/bin/bash
#SBATCH --job-name=sf_pv3_nan_smoke
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/PSEM_v3/AbdAutoPET_2D_Nanfix_Smoke/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/PSEM_v3/AbdAutoPET_2D_Nanfix_Smoke/%x_%j.err

set -euo pipefail

ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/psem_v3_triplet_nanfix
CONDA_ENV=${ROOT}/envs/abdpet
TRAIN_CSV=${ROOT}/csv/AbdAutoPET/Training_A100_Final112_xec.csv
TORCH_HOME=${ROOT}/artifacts/torch
TMPDIR=${ROOT}/artifacts/tmp/${SLURM_JOB_ID}
POSITIVE_COUNTS="59698 43715 34418 30768"
SOURCE_RUN=PSEMv3_TripletContext_Loss3_Delta01_AbdAutoPET_2D_C4_T20_GPU4_B12A4_E1200_RESUME_CKPT400_FP32FIX
RESUME_CHECKPOINT=${ROOT}/Results/PSEM_v3_Triplet_Loss3/AbdAutoPET_2D/${SOURCE_RUN}/checkpoint-475.pth
RUN_NAME=PSEMv3_TripletContext_Loss3_AbdAutoPET_2D_BF16ClipSmoke_Ckpt475_${SLURM_JOB_ID}
OUT_DIR=${ROOT}/Results/PSEM_v3_Triplet_Loss3/_smoke/AbdAutoPET_2D/${RUN_NAME}
RANK_LOG_DIR=${OUT_DIR}/rank_logs

mkdir -p "${OUT_DIR}" "${RANK_LOG_DIR}" "${TMPDIR}"
cd "${WORKDIR}"
git rev-parse HEAD
git status --short
sha256sum VQ/lpips.py engine_pretrain_psem_triplet_loss3.py main_pretrain_psem_triplet_loss3.py util/misc.py
export PATH=${CONDA_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_HOME TMPDIR
PORT_NUM=${MASTER_PORT:-$((10000 + RANDOM % 50000))}

test -x "${CONDA_ENV}/bin/python"
test -r "${TRAIN_CSV}"
test -r "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
if [[ ! -r "${RESUME_CHECKPOINT}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}" >&2
  exit 2
fi
python -c "import os, torch; expected=4; count=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('torch_cuda=', torch.__version__, torch.version.cuda, 'count=', count); assert torch.cuda.is_available() and count == expected; x=torch.ones(1, device='cuda'); assert x.item() == 1"
python -c "import torch; c=torch.load('${RESUME_CHECKPOINT}', map_location='cpu'); assert int(c['epoch']) == 475; assert all(k in c for k in ('model', 'optimizer', 'scaler', 'args')); print('resume checkpoint metadata validated')"
python -c "import torch; from VQ.lpips import normalize_tensor; x=torch.zeros(2,4,3,3,device='cuda',dtype=torch.float16); y=normalize_tensor(x); assert y.dtype == torch.float32 and torch.isfinite(y).all(); print('LPIPS FP32 normalization validated')"
PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_psem_numeric_stability.py'

python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=${PORT_NUM} \
  --log_dir "${RANK_LOG_DIR}" --tee 3 \
  main_pretrain_psem_triplet_loss3.py \
  --batch_size 12 --accum_iter 4 --model mae_vit_base_patch16-LA --mask_ratio 1.0 \
  --epochs 477 --warmup_epochs 60 --blr 1e-4 --weight_decay 0.05 \
  --input_size 224 --num_classes 4 --arch_version v11 \
  --training_version v01 --token_factor 20 --loss_version L2-LPIPS \
  --positive_roi_loss_weight 0.25 --delta_loss_weight 0.1 \
  --roi_positive_sample_counts ${POSITIVE_COUNTS} --roi_frequency_alpha 0.5 \
  --roi_max_weight_ratio 4.0 --lpips_loss_weight 1.0 \
  --intensity_norm per_sample --max_train_samples 512 \
  --amp_dtype bf16 --clip_grad 1.0 --finite_check_interval 1 --deterministic \
  --resume "${RESUME_CHECKPOINT}" --data_path "${TRAIN_CSV}" \
  --output_dir "${OUT_DIR}/" --log_dir "${OUT_DIR}/" \
  --num_workers 4 --save_freq 1 > "${OUT_DIR}/Result.txt" 2>&1

python tools/validate_psem_v3_triplet_smoke.py \
  --run-dir "${OUT_DIR}" --num-foreground-classes 4 --expected-epoch 476
