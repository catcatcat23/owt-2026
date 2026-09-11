#!/usr/bin/env bash
set -euo pipefail
cd "${EXPERIMENT_WORKDIR:?}"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=2D
source scripts/orgslot/arm_f_environment.sh
case "${MAE_RESUME_SCOPE:-encoder_decoder}" in
  encoder) SOURCE_JOB=134494; SOURCE_SCOPE=encoder; RUN_SCOPE=encoder ;;
  encoder_decoder) SOURCE_JOB=133897; SOURCE_SCOPE=encoder_decoder; RUN_SCOPE=encdec ;;
  *) echo 'Unsupported MAE_RESUME_SCOPE' >&2; exit 2 ;;
esac
export MAE_INIT_SCOPE=$SOURCE_SCOPE
export RUN_NAME=ArmE_MAE_${RUN_SCOPE}_resume500_${SLURM_JOB_ID}
export OUTPUT_DIR=${EXPERIMENT_WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
export SLOT_HEAD_TYPE=arm_e_multiscale_query MICRO_BATCH=8 ACCUM_ITER=6
export AMP_DTYPE=fp16 CLIP_GRAD=0 SEGMENTATION_UNIT=slab
export NO_DDP_BUFFER_BROADCAST=1
export RESUME_CHECKPOINT=/gpfs/work/aac/sifansong/worktrees/orgslot_arm_e_mae_20260908/Results/OrganSlotBank/Common8/WORD_2D/OrgSlot_WORD07072_ROI20_ArmE_MAE_${SOURCE_SCOPE}_${SOURCE_JOB}/checkpoint-500.pth
export MAE_INIT_CHECKPOINT="" WORKERS=6 SAVE_FREQ=25
export TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=WARN
printf 'Runtime=%s commit=%s source_job=%s checkpoint=%s\n' "$EXPERIMENT_WORKDIR" "${EXPERIMENT_COMMIT:?}" "$SOURCE_JOB" "$RESUME_CHECKPOINT"
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT" --roi_index "$ROI_INDEX"
"$PYTHON" - "$RESUME_CHECKPOINT" "$SOURCE_SCOPE" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu')
assert c['epoch'] == 500 and c['optimizer']['state'] and c['scaler']
assert c['args'].slot_head_type == 'arm_e_multiscale_query'
assert c['args'].batch_size == 8 and c['args'].accum_iter == 6
assert c['args'].mae_init_scope == sys.argv[2]
assert all(torch.isfinite(v).all() for v in c['model'].values() if v.is_floating_point())
assert all(torch.isfinite(v).all() for state in c['optimizer']['state'].values()
           for v in state.values() if torch.is_tensor(v) and v.is_floating_point())
print('checkpoint500 finite; optimizer and GradScaler present; resume epoch501')
PY
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -s "$OUTPUT_DIR/checkpoint-802.pth"
