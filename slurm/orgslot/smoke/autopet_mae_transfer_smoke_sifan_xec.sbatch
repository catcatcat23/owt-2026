#!/bin/bash
#SBATCH --job-name=xec_ap_mae_smoke
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/AutoPET_MAE/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/AutoPET_MAE/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/orgslot_autopet_mae
PYTHON=${ROOT}/envs/abdpet/bin/python
TRAIN_CSV=${ROOT}/csv/AbdAutoPET/Training_A100_Final112_xec.csv
OUTPUT_DIR=${ROOT}/Results/AutoPET_MAE/_smoke/MAE_LA6D8_224_${SLURM_JOB_ID}

cd "${WORKDIR}"
test -x "${PYTHON}"
test -r "${TRAIN_CSV}"
nvidia-smi -L
"${PYTHON}" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert torch.cuda.is_bf16_supported()"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
MASTER_PORT=$((21000 + SLURM_JOB_ID % 20000))
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  main_pretrain_autopet_mae.py \
  --data_path "${TRAIN_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --input_size 224 \
  --mask_ratio 0.75 \
  --batch_size 2 \
  --accum_iter 1 \
  --max_optimizer_updates 2 \
  --warmup_updates 1 \
  --blr 1e-4 \
  --weight_decay 0.05 \
  --clip_grad 1.0 \
  --save_update_freq 1 \
  --validation_epoch_freq 1 \
  --validation_fraction 0.1 \
  --max_train_samples 8 \
  --max_val_samples 4 \
  --num_workers 2 \
  --seed 0

test -s "${OUTPUT_DIR}/checkpoint-final.pth"
"${PYTHON}" -c "import json,torch; c=torch.load(\"${OUTPUT_DIR}/checkpoint-final.pth\", map_location=\"cpu\"); assert c[\"optimizer_updates\"]==2; rows=[json.loads(x) for x in open(\"${OUTPUT_DIR}/log.txt\")]; assert rows and all(float(x[\"train_loss\"])==float(x[\"train_loss\"]) for x in rows); print(\"AutoPET MAE smoke passed\")"
