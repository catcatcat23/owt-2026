#!/usr/bin/env bash
# Resources and account supplied explicitly by sbatch.
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${EXPERIMENT_COMMIT:?}" "${BACKBONE:?}" "${OUTPUT_DIR:?}"
cd "$EXPERIMENT_WORKDIR"
test "$(git rev-parse HEAD)" = "$EXPERIMENT_COMMIT"
test ! -e "$OUTPUT_DIR"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export PYTHON=${CNN_PYTHON:-/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python}
export LD_LIBRARY_PATH="$(dirname "$PYTHON")/../lib:${LD_LIBRARY_PATH:-}"
echo "commit=$EXPERIMENT_COMMIT runtime=$EXPERIMENT_WORKDIR backbone=$BACKBONE"
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  main_pretrain_autopet_mae.py --backbone "$BACKBONE" \
  --data_path "${CNN_PRETRAIN_CSV:-/gpfs/work/aac/bolinren19/2026-07/Training_A100_Final112.csv}" \
  --output_dir "$OUTPUT_DIR" --batch_size 16 --accum_iter 3 \
  --input_size 224 --mask_ratio .75 --max_optimizer_updates 100000 \
  --warmup_updates 5000 --blr 1e-4 --weight_decay .05 --clip_grad 1 \
  --validation_fraction .1 --validation_epoch_freq 10 --num_workers 6 --seed 0
"$PYTHON" - "$OUTPUT_DIR/checkpoint-final.pth" "$BACKBONE" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu")
assert c['pretraining_complete'] and c['optimizer_updates'] == 100000
assert c['backbone'] == sys.argv[2]
assert all(torch.isfinite(v).all() for v in c['model'].values())
print('CNN pretraining complete and checkpoint finite')
PY
