#!/bin/bash
#SBATCH --job-name=xec_ap_mae100k
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/AutoPET_MAE/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/AutoPET_MAE/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/orgslot_autopet_mae
PYTHON=${ROOT}/envs/abdpet/bin/python
TRAIN_CSV=${ROOT}/csv/AbdAutoPET/Training_A100_Final112_xec.csv
OUTPUT_DIR=${ROOT}/Results/AutoPET_MAE/MAE_LA6D8_224_M75_U100K_seed0

cd "${WORKDIR}"
test -x "${PYTHON}"
test -r "${TRAIN_CSV}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to reuse MAE output: ${OUTPUT_DIR}" >&2
  exit 2
fi
nvidia-smi -L
"${PYTHON}" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4; assert torch.cuda.is_bf16_supported()"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
MASTER_PORT=$((21000 + SLURM_JOB_ID % 20000))
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  main_pretrain_autopet_mae.py \
  --data_path "${TRAIN_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --input_size 224 \
  --mask_ratio 0.75 \
  --batch_size 8 \
  --accum_iter 6 \
  --max_optimizer_updates 100000 \
  --warmup_updates 5000 \
  --blr 1e-4 \
  --weight_decay 0.05 \
  --clip_grad 1.0 \
  --save_update_freq 10000 \
  --validation_epoch_freq 10 \
  --validation_fraction 0.1 \
  --num_workers 6 \
  --seed 0

test -s "${OUTPUT_DIR}/checkpoint-final.pth"
"${PYTHON}" -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-final.pth', map_location='cpu'); assert c['optimizer_updates']==100000; assert c['input_size']==224; assert c['encoder_depth']==6; assert c['decoder_depth']==8; assert c['decoder_dim']==768; print('AutoPET MAE checkpoint validated')"
