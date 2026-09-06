#!/bin/bash
#SBATCH --job-name=sip_word070_head_eval
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiquery/slurm/orgslot/logs/WORD070_UnifiedEval/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiquery/slurm/orgslot/logs/WORD070_UnifiedEval/%x_%j.err

set -euo pipefail

: "${EVAL_WORKDIR:?export EVAL_WORKDIR to the experiment worktree}"
: "${RUN_TAG:?export RUN_TAG without the trailing Slurm job id}"
: "${TRAIN_JOB_ID:?export TRAIN_JOB_ID}"
: "${EXPECTED_HEAD_TYPE:?export EXPECTED_HEAD_TYPE}"

ROOT=/gpfs/work/aac/bolinren19
DATA=${ROOT}/2026-07/DATA/processed/OWT_Common8_07072_NATIVE/WORD
RUN=${RUN_TAG}_${TRAIN_JOB_ID}
CHECKPOINT=${EVAL_WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN}/checkpoint-802.pth
EVAL_ROOT=${EVAL_WORKDIR}/Results/OrganSlotBank/evaluation/Common8/WORD_2D
CAL_OUTPUT=${EVAL_ROOT}/${RUN}_ckpt802_heads_train_calibration
HEAD_OUTPUT=${EVAL_ROOT}/${RUN}_ckpt802_heads_test
RECON_OUTPUT=${EVAL_ROOT}/${RUN}_ckpt802_reconstruction_fixedthr002
CALIBRATION=${CAL_OUTPUT}/selected_thresholds.json
PYTHON=${ROOT}/.conda/envs/abdpet/bin/python

case "${EVAL_WORKDIR}" in
  ${ROOT}/OD_OWT/.worktrees/*) ;;
  *) echo "Refusing unexpected worktree: ${EVAL_WORKDIR}" >&2; exit 2 ;;
esac

test -d "${EVAL_WORKDIR}"
test -r "${CHECKPOINT}"
test -r "${DATA}/csv/WORD_Training_2D_native07072.csv"
test -r "${DATA}/csv/WORD_Test_2D_native07072.csv"
cd "${EVAL_WORKDIR}"

export PATH=${ROOT}/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/.conda/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

"${PYTHON}" - "${CHECKPOINT}" "${EXPECTED_HEAD_TYPE}" <<'PY'
import sys
import torch

path, expected_head = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu")
args = checkpoint.get("args")
assert checkpoint.get("epoch") == 802, checkpoint.get("epoch")
assert getattr(args, "slot_head_type", None) == expected_head
assert list(getattr(args, "expected_spacing", [])) == [0.7, 0.7, 2.0]
assert getattr(args, "input_size", None) == 448
assert getattr(args, "global_crop_size", None) == 448
assert abs(float(getattr(args, "lambda_seg", -1)) - 0.01) < 1e-12
print({
    "epoch": checkpoint["epoch"],
    "slot_head_type": args.slot_head_type,
    "spacing": args.expected_spacing,
    "lambda_seg": args.lambda_seg,
})
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
if complete_key.name == "progress_only":
    ok = True
elif complete_key.name == "complete_test_set":
    ok = result.get("complete_test_set") is True
else:
    ok = result.get("complete_split") is True
ok = ok and progress.get("complete") is True and progress.get("percent") == 100.0
raise SystemExit(0 if ok else 1)
PY
}

if is_complete "${CAL_OUTPUT}/results.json" "${CAL_OUTPUT}/progress.json" progress_only; then
  echo "Calibration already complete; reusing ${CAL_OUTPUT}"
else
  test ! -e "${CAL_OUTPUT}" || { echo "Incomplete calibration output exists: ${CAL_OUTPUT}" >&2; exit 3; }
  "${PYTHON}" -m tools.eval_common8_orgslot_heads \
    --checkpoint "${CHECKPOINT}" \
    --data-csv "${DATA}/csv/WORD_Training_2D_native07072.csv" \
    --output-dir "${CAL_OUTPUT}" \
    --class-map configs/orgslot/common8_offline.json \
    --split train_calibration \
    --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --max-samples 6000 --fusion-mode linear_sqrt \
    --batch-size 4 --workers 8 --min-size 20 --opening-radius 1 --device cuda
fi
test -r "${CALIBRATION}"

if is_complete "${HEAD_OUTPUT}/results.json" "${HEAD_OUTPUT}/progress.json" complete_split; then
  echo "Head test already complete; reusing ${HEAD_OUTPUT}"
else
  test ! -e "${HEAD_OUTPUT}" || { echo "Incomplete head output exists: ${HEAD_OUTPUT}" >&2; exit 4; }
  "${PYTHON}" -m tools.eval_common8_orgslot_heads \
    --checkpoint "${CHECKPOINT}" \
    --data-csv "${DATA}/csv/WORD_Test_2D_native07072.csv" \
    --output-dir "${HEAD_OUTPUT}" \
    --class-map configs/orgslot/common8_offline.json \
    --split test --binary-threshold 0.5 --class-thresholds "${CALIBRATION}" \
    --fusion-mode linear_sqrt --batch-size 4 --workers 8 \
    --min-size 20 --opening-radius 1 --device cuda
fi

if is_complete "${RECON_OUTPUT}/results.json" "${RECON_OUTPUT}/progress.json" complete_test_set; then
  echo "Reconstruction test already complete; reusing ${RECON_OUTPUT}"
else
  test ! -e "${RECON_OUTPUT}" || { echo "Incomplete reconstruction output exists: ${RECON_OUTPUT}" >&2; exit 5; }
  "${PYTHON}" -m tools.eval_common8_orgslot_reconstruction_threshold \
    --method orgslot \
    --checkpoint "${CHECKPOINT}" \
    --test-csv "${DATA}/csv/WORD_Test_2D_native07072.csv" \
    --preprocess-summary "${DATA}/metadata/preprocess_summary.json" \
    --expected-spacing 0.7 0.7 2 \
    --output-dir "${RECON_OUTPUT}" \
    --class-map configs/orgslot/common8_offline.json \
    --fusion-mode linear_sqrt \
    --input-size 448 --global-crop-size 448 \
    --batch-size 4 --workers 8 \
    --threshold 0.02 --min-size 20 --opening-radius 1 \
    --class-ids 1,2,3,4,5,6,7,8 --device cuda
fi

"${PYTHON}" tools/validate_word_orgslot_resolution_eval.py \
  --eval-dir "${RECON_OUTPUT}" --expected-input-size 448 --expected-global-crop 448 \
  --expected-spacing 0.7 0.7 2 --expected-cases 24

echo "Unified evaluation complete for ${RUN}"
