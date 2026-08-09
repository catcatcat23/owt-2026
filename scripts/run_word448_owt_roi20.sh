#!/usr/bin/env bash
set -euo pipefail

REPO=/home/Anteng/OD_OWT_orgslot
PYTHON=/home/Anteng/miniconda3/envs/abdpet/bin/python
OUTPUT=${REPO}/Results/OWTLegacy/Common8/WORD_2D/OWT_WORD112_input448_roi20_legacyTG6_focusunion_eb192_u118800
GPU_LIST=${CUDA_VISIBLE_DEVICES:-0,1}
MASTER_PORT=${MASTER_PORT:-25842}

cd "${REPO}"
mkdir -p "${OUTPUT}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=2 --master_port="${MASTER_PORT}" \
  main_pretrain_owt_common8_a100.py \
  --batch_size 8 --accum_iter 12 \
  --input_size 448 --global_crop_size 448 \
  --roi_crop_size 384 --roi_center_jitter 0.1 \
  --organ_roi_aug --organ_roi_probability 0.2 \
  --focus_class_ids 4,5,6 \
  --roi_index /mnt/DATA-4/anteng/processed/OWT_Common8_112_NATIVE/WORD/metadata/small_organ_roi_index.json \
  --max_optimizer_updates 118800 --warmup_updates 5940 \
  --blr 1e-4 --weight_decay 0.05 --token_factor 20 \
  --loss_version L2-LPIPS --lambda_lpips 1.0 \
  --lpips_state /mnt/DATA-4/anteng/pretrained/owt_lpips_vgg16.pth \
  --lambda_seg 0 --tgr_mode legacy_batch \
  --data_path /mnt/DATA-4/anteng/processed/OWT_Common8_112_NATIVE/WORD/csv/WORD_Training_2D_native112.csv \
  --val_data_path /mnt/DATA-4/anteng/processed/OWT_Common8_112_NATIVE/WORD/csv/WORD_Test_2D_native112.csv \
  --preprocess_summary /mnt/DATA-4/anteng/processed/OWT_Common8_112_NATIVE/WORD/metadata/preprocess_summary.json \
  --visibility_config configs/orgslot/common8_offline.json \
  --output_dir "${OUTPUT}" --log_dir "${OUTPUT}" \
  --save_freq 100 --num_workers 10 --print_freq 20 --seed 0
