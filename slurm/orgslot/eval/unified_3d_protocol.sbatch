#!/usr/bin/env bash
#SBATCH --job-name=orgslot-3d-unified
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPERIMENT_COMMIT:?}" "${CHECKPOINT:?}"
: "${PROCESSED_ROOT:?}" "${PYTHON:?}" "${OUTPUT_ROOT:?}" "${EVAL_STAGE:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(git rev-parse HEAD)" = "$EXPERIMENT_COMMIT"
export PYTHONPATH="$EXPERIMENT_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
args=(--checkpoint "$CHECKPOINT" --test-reference-csv "$PROCESSED_ROOT/csv/WORD_Test_2D_native07072.csv"
      --preprocess-summary "$PROCESSED_ROOT/metadata/preprocess_summary.json"
      --output-dir "$OUTPUT_ROOT/$EVAL_STAGE" --mode "$EVAL_STAGE" --workers 4)
case "$EVAL_STAGE" in
  calibrate)
    args+=(--data-csv "$PROCESSED_ROOT/csv/WORD_Training_2D_native07072.csv" --expected-cases 96);;
  head)
    args+=(--data-csv "$PROCESSED_ROOT/csv/WORD_Test_2D_native07072.csv" --expected-cases 24
           --class-thresholds "$OUTPUT_ROOT/calibrate/selected_thresholds.json");;
  reconstruction)
    args+=(--data-csv "$PROCESSED_ROOT/csv/WORD_Test_2D_native07072.csv" --expected-cases 24);;
  *) exit 2;;
esac
"$PYTHON" -u tools/eval_common8_orgslot_unified_3d.py "${args[@]}"
