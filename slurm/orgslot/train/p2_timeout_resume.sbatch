#!/usr/bin/env bash
# Submit with afterany of PREVIOUS_JOB. Preserve the existing experiment revision.
set -euo pipefail
: "${PREVIOUS_JOB:?}" "${PINNED_WORKDIR:?}" "${ARM_F_HEAD_TYPE:?}"
state=$(sacct -n -X -j "$PREVIOUS_JOB" -o State | awk 'NF {print $1; exit}')
case "$state" in COMPLETED|TIMEOUT) ;; *) echo "Refusing continuation after $state" >&2; exit 2;; esac
expected=efcef0739b581ec49c969c312d07c97574308cbb
test "$(git -C "$PINNED_WORKDIR" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$PINNED_WORKDIR" status --porcelain --untracked-files=no)"
export EXPERIMENT_WORKDIR=$PINNED_WORKDIR ARM_F_TARGET=anteng-xec ARM_F_DIMENSION=2D
export ARM_F_COMMIT=$expected ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3 BACKGROUND_REDUCTION=topk
cd "$EXPERIMENT_WORKDIR"
source scripts/orgslot/arm_f_environment.sh
export RUN_NAME=ArmF_2D_${PREVIOUS_JOB}
export OUTPUT_DIR=$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME
export RESUME_CHECKPOINT
RESUME_CHECKPOINT=$("$PYTHON" - "$OUTPUT_DIR" "$ARM_F_HEAD_TYPE" <<'PY'
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
paths = sorted(root.glob("checkpoint-*.pth"), key=lambda p: int(p.stem.split("-")[-1]), reverse=True)
for path in paths:
    try:
        c = torch.load(path, map_location="cpu")
    except Exception as exc:
        print("Unreadable checkpoint: %s (%s)" % (path, exc), file=sys.stderr)
        continue
    a = c["args"]
    assert a.slot_head_type == sys.argv[2]
    assert a.batch_size == 16 and a.accum_iter == 3
    assert abs(a.lambda_seg - 0.01) < 1e-12
    assert "optimizer" in c and "scaler" in c and "model" in c
    print("Resume epoch=%s; optimizer/scaler present" % c["epoch"], file=sys.stderr)
    print(path)
    break
else:
    raise RuntimeError("No readable checkpoint; refusing scratch restart")
PY
)
echo "Resume $RESUME_CHECKPOINT with pinned commit $expected"
if [[ "$RESUME_CHECKPOINT" == */checkpoint-802.pth ]]; then
    echo "Final checkpoint already complete; proceed to existing evaluation"
    exit 0
fi
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT" --roi_index "$ROI_INDEX"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
"$PYTHON" - "$OUTPUT_DIR/log.txt" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
assert d["train_optimizer_updates"] >= 118800, d
print("Full update budget verified")
PY
