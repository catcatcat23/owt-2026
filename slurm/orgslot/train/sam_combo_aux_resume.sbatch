#!/usr/bin/env bash
# Resource/account/time options supplied explicitly by sbatch.
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${RESUME_CHECKPOINT:?}" "${ARM_F_COMMIT:?}"
cd "$EXPERIMENT_WORKDIR"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=2D ARM_F_HEAD_TYPE=arm_f_sam_tail
export ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3 ARM_F_LAMBDA_SEG=0.01
export ARM_F_SEG_SUPERVISION=retained BACKGROUND_REDUCTION=topk
export HSAM_SUPERVISION=downsample_gt_soft_prior_aux
export MAE_INIT_SCOPE=encoder MAE_INIT_CHECKPOINT=""
# Existing trainer rejects disabling broadcasts if any buffer is not immutable
# LPIPS shift/scale. Gradient collectives remain enabled.
export NO_DDP_BUFFER_BROADCAST=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
source scripts/orgslot/arm_f_environment.sh
"$PYTHON" - "$RESUME_CHECKPOINT" <<'PY_CHECK'
import sys
import torch

c = torch.load(sys.argv[1], map_location="cpu")
a = vars(c["args"])
expected = {
    "hsam_supervision": "downsample_gt_soft_prior_aux",
    "slot_head_type": "arm_f_sam_tail", "dimension": "2D",
    "mae_init_scope": "encoder", "batch_size": 16, "accum_iter": 3,
    "max_optimizer_updates": 118800, "warmup_updates": 5940,
    "seg_loss_type": "small_organ", "background_reduction": "topk",
    "seg_supervision": "retained", "lambda_seg": .01,
    "amp_dtype": "bf16", "clip_grad": 1.0, "seed": 0,
}
for name, value in expected.items():
    assert a.get(name) == value, (name, a.get(name), value)
assert c["epoch"] == 470, c["epoch"]
assert "scaler" in c and c["optimizer"]["state"]
assert all(torch.isfinite(t).all() for t in c["model"].values()
           if torch.is_tensor(t) and t.is_floating_point())
steps = {int(s["step"]) for s in c["optimizer"]["state"].values() if "step" in s}
assert steps == {69708}, steps
assert all(torch.isfinite(t).all() for s in c["optimizer"]["state"].values()
           for t in s.values() if torch.is_tensor(t) and t.is_floating_point())
print("Verified epoch470 / 69708 updates; resume at epoch471, restore optimizer/scaler",
      flush=True)
PY_CHECK
provenance="$(dirname "$RESUME_CHECKPOINT")/mae_initial_checkpoint_load.json"
test -s "$provenance"
bash slurm/orgslot/train/arm_f.sbatch
# Preserve the original MAE initialization report for the resumed evaluation.
output="$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/ArmF_2D_${SLURM_JOB_ID}"
cp "$provenance" "$output/mae_initial_checkpoint_load.json"
