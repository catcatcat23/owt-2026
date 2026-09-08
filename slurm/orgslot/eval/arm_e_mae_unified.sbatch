#!/bin/bash
#SBATCH --job-name=armE_MAE_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${EVAL_WORKDIR:?}" "${RUN_NAME:?}" "${ACCOUNT_ROOT:?}" "${DATA:?}" "${PYTHON:?}"
cd "${EVAL_WORKDIR}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}"
CHECKPOINT=${EVAL_WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}/checkpoint-802.pth
EVAL_ROOT=${EVAL_WORKDIR}/Results/OrganSlotBank/evaluation/Common8/WORD_2D
CAL_OUTPUT=${EVAL_ROOT}/${RUN_NAME}_ckpt802_heads_train_calibration
HEAD_OUTPUT=${EVAL_ROOT}/${RUN_NAME}_ckpt802_heads_test
RECON_OUTPUT=${EVAL_ROOT}/${RUN_NAME}_ckpt802_reconstruction_fixedthr002
test -s "${CHECKPOINT}"
"${PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
a = c["args"]
assert c["epoch"] == 802 and a.slot_head_type == "arm_e_multiscale_query"
assert list(a.expected_spacing) == [0.7, 0.7, 2.0]
assert a.input_size == 448 and a.global_crop_size == 448
assert a.lambda_seg == 0.01 and a.organ_roi_probability == 0.2
print("Matched Arm E checkpoint", c["epoch"])
PY
"${PYTHON}" -m tools.eval_common8_orgslot_heads \
 --checkpoint "${CHECKPOINT}" --data-csv "${DATA}/csv/WORD_Training_2D_native07072.csv" \
 --output-dir "${CAL_OUTPUT}" --class-map configs/orgslot/common8_offline.json \
 --split train_calibration --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
 --max-samples 6000 --fusion-mode linear_sqrt --batch-size 4 --workers 8 --min-size 20 --opening-radius 1 --device cuda
"${PYTHON}" -m tools.eval_common8_orgslot_heads \
 --checkpoint "${CHECKPOINT}" --data-csv "${DATA}/csv/WORD_Test_2D_native07072.csv" \
 --output-dir "${HEAD_OUTPUT}" --class-map configs/orgslot/common8_offline.json \
 --split test --binary-threshold 0.5 --class-thresholds "${CAL_OUTPUT}/selected_thresholds.json" \
 --fusion-mode linear_sqrt --batch-size 4 --workers 8 --min-size 20 --opening-radius 1 --device cuda
"${PYTHON}" -m tools.eval_common8_orgslot_reconstruction_threshold \
 --method orgslot --checkpoint "${CHECKPOINT}" --test-csv "${DATA}/csv/WORD_Test_2D_native07072.csv" \
 --preprocess-summary "${DATA}/metadata/preprocess_summary.json" --expected-spacing 0.7 0.7 2 \
 --output-dir "${RECON_OUTPUT}" --class-map configs/orgslot/common8_offline.json --fusion-mode linear_sqrt \
 --input-size 448 --global-crop-size 448 --batch-size 4 --workers 8 --threshold 0.02 \
 --min-size 20 --opening-radius 1 --class-ids 1,2,3,4,5,6,7,8 --device cuda
"${PYTHON}" tools/validate_word_orgslot_resolution_eval.py --eval-dir "${RECON_OUTPUT}" \
 --expected-input-size 448 --expected-global-crop 448 --expected-spacing 0.7 0.7 2 --expected-cases 24
