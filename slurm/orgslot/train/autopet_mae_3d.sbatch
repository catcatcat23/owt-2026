#!/usr/bin/env bash
# Resource options, account, QoS, log paths, and maximum wall time are supplied
# explicitly by sbatch for the selected cluster/account.
set -euo pipefail

cd "${EXPERIMENT_WORKDIR:?}"
test "$(git rev-parse HEAD)" = "${EXPERIMENT_COMMIT:?}"
test -z "$(git status --porcelain --untracked-files=normal)"
test -x "${PYTHON:?}"
test -r "${TRAIN_CSV:?}"
if [[ -e "${OUTPUT_DIR:?}" ]]; then
  printf 'Refusing to reuse MAE output: %s\n' "${OUTPUT_DIR}" >&2
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
  --dimension 3D \
  --frames 4 \
  --temporal_patch_size 1 \
  --data_path "${TRAIN_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --input_size 224 \
  --mask_ratio 0.75 \
  --batch_size 4 \
  --accum_iter 12 \
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
"${PYTHON}" -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-final.pth', map_location='cpu'); assert c['optimizer_updates']==100000; assert c['input_size']==224; assert c['dimension']=='3D'; assert c['frames']==4; assert c['temporal_patch_size']==1; assert c['encoder_depth']==6; assert c['decoder_depth']==8; assert c['decoder_dim']==768; print('AutoPET 3D MAE checkpoint validated')"
