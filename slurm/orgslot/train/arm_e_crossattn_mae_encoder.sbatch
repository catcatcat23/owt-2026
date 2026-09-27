#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}"
cd "$EXPERIMENT_WORKDIR"
export ARM_F_TARGET=sifan-xec ARM_F_DIMENSION=2D
export ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3 BACKGROUND_REDUCTION=topk
source scripts/orgslot/arm_f_environment.sh
export SLOT_HEAD_TYPE=arm_e_multiscale_query QUERY_REFINEMENT=cross_attn
export MAE_INIT_SCOPE=encoder RESUME_CHECKPOINT=""
export MAE_INIT_CHECKPOINT=/gpfs/work/aac/sifansong/Results/AutoPET_MAE/MAE_LA6D8_224_M75_U100K_seed0/checkpoint-final.pth
export RUN_NAME=ArmE_CrossAttn_MAEenc_${SLURM_JOB_ID}
export OUTPUT_DIR=$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME
printf '0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  %s\n' "$MAE_INIT_CHECKPOINT" | sha256sum -c -
"$PYTHON" tools/validate_word070_dataset_identity.py --processed_root "$PROCESSED_ROOT" --roi_index "$ROI_INDEX"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -s "$OUTPUT_DIR/checkpoint-802.pth"
test -s "$OUTPUT_DIR/mae_initial_checkpoint_load.json"
