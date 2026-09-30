#!/usr/bin/env bash
#SBATCH --job-name=Ecross_replace5_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${REPLACEMENT_DATA:?}" "${OUTPUT_ROOT:?}"
cd "$EXPERIMENT_WORKDIR"
export PYTHONNOUSERSITE=1
PYTHON=/gpfs/work/aac/sifansong/envs/abdpet/bin/python
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
OLD_ROOT=/gpfs/work/aac/sifansong/worktrees/e_cross_mae_9791e87/Results/OrganSlotBank
OLD_RUN=$OLD_ROOT/Common8/WORD_2D/ArmE_CrossAttn_MAEenc_148833
OLD_EVAL=$OLD_ROOT/evaluation/Common8/WORD_2D/ArmE_CrossAttn_MAEenc_148833_ckpt802
THRESHOLDS=${OLD_EVAL}_heads_train_calibration/selected_thresholds.json
COMMON=(--plan configs/orgslot/word_replacement5_seed42.json
 --old-data /gpfs/work/aac/sifansong/data/OWT_Common8_07072_NATIVE/WORD
 --old-run "$OLD_RUN" --replacement-data "$REPLACEMENT_DATA" --output "$OUTPUT_ROOT"
 --thresholds "$THRESHOLDS" --old-results "${OLD_EVAL}_heads_test/results.json")
"$PYTHON" tools/word_replacement_eval.py prepare "${COMMON[@]}"
"$PYTHON" - "$OLD_RUN/checkpoint-802.pth" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
a = c['args']
assert c['epoch'] == 802 and a.slot_head_type == 'arm_e_multiscale_query'
assert a.query_refinement == 'cross_attn' and a.mae_init_scope == 'encoder'
assert a.input_size == 448 and list(a.expected_spacing) == [0.7,0.7,2.0]
print('Verified original E cross-attention MAE encoder checkpoint')
PY
"$PYTHON" -m tools.eval_common8_orgslot_heads \
 --checkpoint "$OLD_RUN/checkpoint-802.pth" --data-csv "$OUTPUT_ROOT/test.csv" \
 --output-dir "$OUTPUT_ROOT/heads" --class-map configs/orgslot/common8_offline.json \
 --split test --binary-threshold 0.5 --class-thresholds "$THRESHOLDS" \
 --fusion-mode linear_sqrt --batch-size 4 --workers 4 --min-size 20 --opening-radius 1 --device cuda
"$PYTHON" tools/word_replacement_eval.py compare "${COMMON[@]}"
