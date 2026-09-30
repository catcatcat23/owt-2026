#!/usr/bin/env bash
#SBATCH --job-name=Ecross_MAE_official96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}" "${NEW_DATA_ROOT:?}"
cd "$EXPERIMENT_WORKDIR"
export ARM_F_TARGET=bolin-sip ARM_F_DIMENSION=2D
export ARM_F_MICRO_BATCH=16 ARM_F_ACCUM_ITER=3 BACKGROUND_REDUCTION=topk
export ARM_F_LAMBDA_SEG=0.01 ARM_F_SEG_SUPERVISION=retained
source scripts/orgslot/arm_f_environment.sh
export PROCESSED_ROOT=$NEW_DATA_ROOT/WORD
export ROI_INDEX=$PROCESSED_ROOT/metadata/small_organ_roi_index.json
export TRAIN_CSV=$PROCESSED_ROOT/csv/WORD_Training_2D_native07072.csv
export VAL_CSV=$PROCESSED_ROOT/csv/WORD_Validation_2D_native07072.csv
export SLOT_HEAD_TYPE=arm_e_multiscale_query QUERY_REFINEMENT=cross_attn
export MAE_INIT_SCOPE=encoder RESUME_CHECKPOINT=""
export MAE_INIT_CHECKPOINT=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_autopet_mae/Results/AutoPET_MAE/MAE_LA6D8_224_M75_U100K_seed0/checkpoint-final.pth
export RUN_NAME=ArmE_CrossAttn_MAEenc_Official96_${SLURM_JOB_ID}
export OUTPUT_DIR=$EXPERIMENT_WORKDIR/Results/OrganSlotBank/Common8/WORD_2D/$RUN_NAME
test -s "$PROCESSED_ROOT/metadata/READY.json"
printf '0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  %s\n' "$MAE_INIT_CHECKPOINT" | sha256sum -c -
"$PYTHON" tools/prepare_word_official96.py validate --plan configs/orgslot/word_official96_seed42.json --output "$NEW_DATA_ROOT"
bash scripts/orgslot/run_word_112_448_a100.sh roi20
"$PYTHON" tools/word_official96_final_checkpoint.py "$OUTPUT_DIR"
