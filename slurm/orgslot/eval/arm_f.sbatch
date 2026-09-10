#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
source scripts/orgslot/arm_f_environment.sh
RUN_NAME=ArmF_${DIMENSION}_${TRAIN_JOB_ID:?}
TRAIN_OUTPUT=${EXPERIMENT_WORKDIR}/Results/OrganSlotBank/Common8/WORD_${DIMENSION}/${RUN_NAME}
CHECKPOINT=$(find "$TRAIN_OUTPUT" -maxdepth 1 -name 'checkpoint-*.pth' | sort -V | tail -1)
test -s "$CHECKPOINT"
OUTPUT=${EXPERIMENT_WORKDIR}/Results/OrganSlotBank/evaluation/${RUN_NAME}
if [[ "$DIMENSION" == 2D ]]; then
  "$PYTHON" -m tools.eval_common8_orgslot_heads --checkpoint "$CHECKPOINT" \
    --data-csv "$PROCESSED_ROOT/csv/WORD_Test_2D_native07072.csv" \
    --output-dir "$OUTPUT" --split test --input-size 448 --global-crop-size 448 \
    --batch-size 1 --workers 4 --binary-threshold 0.5 --min-size 20 --opening-radius 1 \
    --fusion-mode linear_sqrt
else
  "$PYTHON" -m tools.eval_common8_orgslot_heads_3d --checkpoint "$CHECKPOINT" \
    --test-csv "$PROCESSED_ROOT/csv/WORD_Test_2D_native07072.csv" \
    --preprocess-summary "$PROCESSED_ROOT/metadata/preprocess_summary.json" \
    --expected-spacing 0.7 0.7 2 --expected-cases 24 --output-dir "$OUTPUT" \
    --input-size 448 --global-crop-size 448 --fix-frame 4 --temp-stride 1 \
    --batch-size 1 --workers 4 --binary-threshold 0.5 --min-component-voxels 20 \
    --opening-radius-mm 0 --nsd-tolerance-mm 3 --fusion-mode linear_sqrt --device cuda
fi
