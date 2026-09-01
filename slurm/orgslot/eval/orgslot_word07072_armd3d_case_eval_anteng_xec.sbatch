#!/bin/bash
#SBATCH --job-name=xec_os070_3dD_eval
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD070_ArmD3D_Eval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/antengcai23/slurm/logs/OrganSlot/WORD070_ArmD3D_Eval/%x_%j.err

set -euo pipefail

ROOT=/gpfs/work/aac/antengcai23
WORKDIR=${ROOT}/worktrees/orgslot_querymask_3d
TRAIN_JOB_ID=${TRAIN_JOB_ID:?submit with --export=ALL,TRAIN_JOB_ID=<job_id>}
RUN_TAG=OrgSlot_WORD07072_ROI20_RetainedMultiQueryDot3D_FixFr4_SmallOrgan_L0.01_BF16_Clip1
RUN_NAME=${RUN_TAG}_${TRAIN_JOB_ID}
CHECKPOINT=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_3D/${RUN_NAME}/checkpoint-199.pth
DATA=${ROOT}/data/OWT_Common8_07072_NATIVE/WORD
OUTPUT=${WORKDIR}/Results/OrganSlotBank/evaluation/Common8/WORD_3D/${RUN_NAME}_ckpt199_head_case3d_fixed05
PYTHON=${ROOT}/envs/abdpet/bin/python

cd "${WORKDIR}"
export PATH=${ROOT}/envs/abdpet/bin:${PATH}
export PYTHONNOUSERSITE=1

test -r "${CHECKPOINT}"
test -r "${DATA}/csv/WORD_Test_2D_native07072.csv"
test ! -e "${OUTPUT}"
"${PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
args = checkpoint["args"]
assert int(checkpoint["epoch"]) == 199
assert getattr(args, "dimension") == "3D"
assert int(getattr(args, "fix_frame")) == 4
assert int(getattr(args, "temp_stride")) == 1
assert list(getattr(args, "expected_spacing")) == [0.7, 0.7, 2.0]
assert getattr(args, "slot_head_type") == "multi_query_dot"
assert getattr(args, "seg_loss_type") == "small_organ"
assert float(getattr(args, "lambda_seg")) == 0.01
PY

nvidia-smi -L
"${PYTHON}" -m tools.eval_common8_orgslot_heads_3d \
  --checkpoint "${CHECKPOINT}" \
  --test-csv "${DATA}/csv/WORD_Test_2D_native07072.csv" \
  --preprocess-summary "${DATA}/metadata/preprocess_summary.json" \
  --expected-spacing 0.7 0.7 2 --expected-cases 24 \
  --output-dir "${OUTPUT}" \
  --class-map configs/orgslot/common8_offline.json \
  --input-size 448 --global-crop-size 448 \
  --fix-frame 4 --temp-stride 1 \
  --batch-size 1 --workers 8 --binary-threshold 0.5 \
  --min-component-voxels 20 --opening-radius-mm 0 \
  --nsd-tolerance-mm 3 --fusion-mode linear_sqrt --device cuda

"${PYTHON}" - "${OUTPUT}" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
result = json.loads((output / "results.json").read_text())
progress = json.loads((output / "progress.json").read_text())
assert result["complete_test_set"] is True and result["cases"] == 24
assert progress["complete"] is True and progress["percent"] == 100.0
assert len((output / "per_case.csv").read_text().splitlines()) == 24 * 8 + 1
print("formal true-3D Arm D evaluation validated")
PY
