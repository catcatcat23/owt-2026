#!/usr/bin/env bash
#SBATCH --job-name=Ecross_official96_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${NEW_DATA_ROOT:?}" "${TRAIN_JOB_ID:?}"
cd "$EXPERIMENT_WORKDIR"
if [[ -n "${EXPECTED_COMMIT:-}" ]]; then
 test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
fi
export ARM_F_TARGET=bolin-sip ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
DATA=$NEW_DATA_ROOT/WORD
RUN_NAME=ArmE_CrossAttn_MAEenc_Official96_${TRAIN_JOB_ID}
RUN=${TRAIN_RUN:-$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME}
CHECKPOINT=$("$PYTHON" tools/word_official96_final_checkpoint.py "$RUN")
"$PYTHON" tools/prepare_word_official96.py validate --plan "${SPLIT_PLAN:-configs/orgslot/word_official96_seed42.json}" --output "$NEW_DATA_ROOT"
EVAL_ROOT=${EVAL_ROOT:-$EXPERIMENT_WORKDIR/Results/OrganSlotBank/evaluation/Common8/WORD_2D}
CAL_OUTPUT=$EVAL_ROOT/${RUN_NAME}_heads_validation
HEAD_OUTPUT=$EVAL_ROOT/${RUN_NAME}_heads_test
RECON_OUTPUT=$EVAL_ROOT/${RUN_NAME}_reconstruction_fixedthr002
"$PYTHON" -m tools.eval_common8_orgslot_heads \
 --checkpoint "$CHECKPOINT" --data-csv "$DATA/csv/WORD_Validation_2D_native07072.csv" \
 --output-dir "$CAL_OUTPUT" --class-map configs/orgslot/common8_offline.json \
 --split validation --threshold-sweep 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
 --fusion-mode linear_sqrt --batch-size 4 --workers 8 --min-size 20 --opening-radius 1 --device cuda
"$PYTHON" -m tools.eval_common8_orgslot_heads \
 --checkpoint "$CHECKPOINT" --data-csv "$DATA/csv/WORD_Test_2D_native07072.csv" \
 --output-dir "$HEAD_OUTPUT" --class-map configs/orgslot/common8_offline.json \
 --split test --binary-threshold 0.5 --class-thresholds "$CAL_OUTPUT/selected_thresholds.json" \
 --fusion-mode linear_sqrt --batch-size 4 --workers 8 --min-size 20 --opening-radius 1 --device cuda
"$PYTHON" -m tools.eval_common8_orgslot_reconstruction_threshold \
 --method orgslot --checkpoint "$CHECKPOINT" --test-csv "$DATA/csv/WORD_Test_2D_native07072.csv" \
 --preprocess-summary "$DATA/metadata/preprocess_summary.json" --expected-spacing 0.7 0.7 2 \
 --output-dir "$RECON_OUTPUT" --class-map configs/orgslot/common8_offline.json --fusion-mode linear_sqrt \
 --input-size 448 --global-crop-size 448 --batch-size 4 --workers 8 --threshold 0.02 \
 --min-size 20 --opening-radius 1 --class-ids 1,2,3,4,5,6,7,8 --device cuda
"$PYTHON" tools/validate_word_orgslot_resolution_eval.py --eval-dir "$RECON_OUTPUT" \
 --expected-input-size 448 --expected-global-crop 448 --expected-spacing 0.7 0.7 2 --expected-cases 24
