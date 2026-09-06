#!/bin/bash
#SBATCH --job-name=xec_os070_maeE_eval
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_AutoPETMAE/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_AutoPETMAE/%x_%j.err

set -euo pipefail

ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/orgslot_autopet_mae
DATA=${ROOT}/data/OWT_Common8_07072_NATIVE/WORD
RUN=OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncoderOnly_128402
CHECKPOINT=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN}/checkpoint-802.pth
EVAL_ROOT=${WORKDIR}/Results/OrganSlotBank/evaluation/Common8/WORD_2D
CAL_OUTPUT=${EVAL_ROOT}/${RUN}_ckpt802_heads_train_calibration
HEAD_OUTPUT=${EVAL_ROOT}/${RUN}_ckpt802_heads_test
RECON_OUTPUT=${EVAL_ROOT}/${RUN}_ckpt802_reconstruction_fixedthr002
CALIBRATION=${CAL_OUTPUT}/selected_thresholds.json
PYTHON=${ROOT}/envs/abdpet/bin/python

test -x "${PYTHON}"
test -r "${CHECKPOINT}"
test -r "${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN}/mae_initial_checkpoint_load.json"
test -r "${DATA}/csv/WORD_Training_2D_native07072.csv"
test -r "${DATA}/csv/WORD_Test_2D_native07072.csv"
test -r "${DATA}/metadata/preprocess_summary.json"
cd "${WORKDIR}"
export PATH=${ROOT}/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/envs/abdpet/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

"${PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
args = checkpoint["args"]
assert checkpoint.get("epoch") == 802
assert args.slot_head_type == "multiscale_conv"
assert args.mae_init_scope == "encoder"
assert list(args.expected_spacing) == [0.7, 0.7, 2.0]
assert args.input_size == 448 and args.global_crop_size == 448
assert abs(float(args.lambda_seg) - 0.01) < 1e-12
print({"epoch": checkpoint["epoch"], "head": args.slot_head_type,
       "mae_init_scope": args.mae_init_scope})
PY

nvidia-smi -L

"${PYTHON}" -m tools.eval_common8_orgslot_heads \
  --checkpoint "${CHECKPOINT}" \
  --data-csv "${DATA}/csv/WORD_Training_2D_native07072.csv" \
  --output-dir "${CAL_OUTPUT}" \
  --class-map configs/orgslot/common8_offline.json \
  --split train_calibration \
  --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --max-samples 6000 --fusion-mode linear_sqrt \
  --batch-size 4 --workers 8 --min-size 20 --opening-radius 1 --device cuda

test -r "${CALIBRATION}"
"${PYTHON}" -m tools.eval_common8_orgslot_heads \
  --checkpoint "${CHECKPOINT}" \
  --data-csv "${DATA}/csv/WORD_Test_2D_native07072.csv" \
  --output-dir "${HEAD_OUTPUT}" \
  --class-map configs/orgslot/common8_offline.json \
  --split test --binary-threshold 0.5 --class-thresholds "${CALIBRATION}" \
  --fusion-mode linear_sqrt --batch-size 4 --workers 8 \
  --min-size 20 --opening-radius 1 --device cuda

"${PYTHON}" -m tools.eval_common8_orgslot_reconstruction_threshold \
  --method orgslot --checkpoint "${CHECKPOINT}" \
  --test-csv "${DATA}/csv/WORD_Test_2D_native07072.csv" \
  --preprocess-summary "${DATA}/metadata/preprocess_summary.json" \
  --expected-spacing 0.7 0.7 2 --output-dir "${RECON_OUTPUT}" \
  --class-map configs/orgslot/common8_offline.json --fusion-mode linear_sqrt \
  --input-size 448 --global-crop-size 448 --batch-size 4 --workers 8 \
  --threshold 0.02 --min-size 20 --opening-radius 1 \
  --class-ids 1,2,3,4,5,6,7,8 --device cuda

"${PYTHON}" tools/validate_word_orgslot_resolution_eval.py \
  --eval-dir "${RECON_OUTPUT}" --expected-input-size 448 \
  --expected-global-crop 448 --expected-spacing 0.7 0.7 2 \
  --expected-cases 24

echo "B1 encoder-only unified evaluation complete: ${RUN}"
