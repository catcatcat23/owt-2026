#!/bin/bash
#SBATCH --job-name=sip_focal_c802_eval
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_head_ab/slurm/orgslot/logs/WORD448_FocalCkpt802/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_head_ab/slurm/orgslot/logs/WORD448_FocalCkpt802/%x_%j.err

set -euo pipefail

ROOT=/gpfs/work/aac/bolinren19
WORKDIR=${ROOT}/OD_OWT/.worktrees/orgslot_head_ab
DATA=${ROOT}/2026-07/DATA/processed/OWT_Common8_112_NATIVE/WORD
TRAIN_RUN=OrgSlot_WORD448_ROI20_RetainedMultiConv_Focal_L0.1_Resume500_Bolin_2413566
RUN=${TRAIN_RUN}_ckpt802
CHECKPOINT=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${TRAIN_RUN}/checkpoint-802.pth
EVAL_ROOT=${WORKDIR}/Results/OrganSlotBank/evaluation/Common8/WORD_2D
CAL_OUTPUT=${EVAL_ROOT}/${RUN}_heads_train_calibration
HEAD_OUTPUT=${EVAL_ROOT}/${RUN}_heads_test
RECON_OUTPUT=${EVAL_ROOT}/${RUN}_reconstruction_fixedthr002
CALIBRATION=${CAL_OUTPUT}/selected_thresholds.json
PYTHON=${ROOT}/.conda/envs/abdpet/bin/python
cd "${WORKDIR}"

export PATH=${ROOT}/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/.conda/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

test "$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)" = "c68b5791534321b60bb85bc427d2c197dbcd08e3618e84541a29e94cf50db5e6"
"${PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
args = checkpoint.get("args")
assert checkpoint.get("epoch") == 802
assert getattr(args, "slot_head_type", None) == "multiscale_conv"
assert list(getattr(args, "expected_spacing", [])) == [1.0, 1.0, 2.0]
assert abs(float(getattr(args, "lambda_seg", -1)) - 0.1) < 1e-12
PY
nvidia-smi -L

is_complete() {
  "${PYTHON}" - "$1" "$2" "$3" <<'PY'
import json
import pathlib
import sys

result_path, progress_path, complete_key = map(pathlib.Path, sys.argv[1:])
if not result_path.is_file() or not progress_path.is_file():
    raise SystemExit(1)
result = json.loads(result_path.read_text())
progress = json.loads(progress_path.read_text())
key = complete_key.name
ok = True if key == "progress_only" else result.get(key) is True
ok = ok and progress.get("complete") is True and progress.get("percent") == 100.0
raise SystemExit(0 if ok else 1)
PY
}

if is_complete "${CAL_OUTPUT}/results.json" "${CAL_OUTPUT}/progress.json" progress_only; then
  echo "Calibration already complete"
else
  test ! -e "${CAL_OUTPUT}" || { echo "Incomplete calibration output exists: ${CAL_OUTPUT}" >&2; exit 3; }
  "${PYTHON}" -m tools.eval_common8_orgslot_heads \
    --checkpoint "${CHECKPOINT}" \
    --data-csv "${DATA}/csv/WORD_Training_2D_native112.csv" \
    --output-dir "${CAL_OUTPUT}" \
    --class-map configs/orgslot/common8_offline.json \
    --split train_calibration \
    --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --max-samples 6000 --fusion-mode linear_sqrt \
    --batch-size 4 --workers 8 --min-size 20 --opening-radius 1 --device cuda
fi
test -r "${CALIBRATION}"

if is_complete "${HEAD_OUTPUT}/results.json" "${HEAD_OUTPUT}/progress.json" complete_split; then
  echo "Head test already complete"
else
  test ! -e "${HEAD_OUTPUT}" || { echo "Incomplete head output exists: ${HEAD_OUTPUT}" >&2; exit 4; }
  "${PYTHON}" -m tools.eval_common8_orgslot_heads \
    --checkpoint "${CHECKPOINT}" \
    --data-csv "${DATA}/csv/WORD_Test_2D_native112.csv" \
    --output-dir "${HEAD_OUTPUT}" \
    --class-map configs/orgslot/common8_offline.json \
    --split test --binary-threshold 0.5 --class-thresholds "${CALIBRATION}" \
    --fusion-mode linear_sqrt --batch-size 4 --workers 8 \
    --min-size 20 --opening-radius 1 --device cuda
fi

if is_complete "${RECON_OUTPUT}/results.json" "${RECON_OUTPUT}/progress.json" complete_test_set; then
  echo "Reconstruction test already complete"
else
  test ! -e "${RECON_OUTPUT}" || { echo "Incomplete reconstruction output exists: ${RECON_OUTPUT}" >&2; exit 5; }
  "${PYTHON}" -m tools.eval_common8_orgslot_reconstruction_threshold \
    --method orgslot --checkpoint "${CHECKPOINT}" \
    --test-csv "${DATA}/csv/WORD_Test_2D_native112.csv" \
    --preprocess-summary "${DATA}/metadata/preprocess_summary.json" \
    --expected-spacing 1 1 2 --output-dir "${RECON_OUTPUT}" \
    --class-map configs/orgslot/common8_offline.json --fusion-mode linear_sqrt \
    --input-size 448 --global-crop-size 448 --batch-size 4 --workers 8 \
    --threshold 0.02 --min-size 20 --opening-radius 1 \
    --class-ids 1,2,3,4,5,6,7,8 --device cuda
fi

"${PYTHON}" tools/validate_word_orgslot_resolution_eval.py \
  --eval-dir "${RECON_OUTPUT}" --expected-input-size 448 --expected-global-crop 448 \
  --expected-spacing 1 1 2 --expected-cases 24

echo "Unified Focal checkpoint-802 evaluation complete"
