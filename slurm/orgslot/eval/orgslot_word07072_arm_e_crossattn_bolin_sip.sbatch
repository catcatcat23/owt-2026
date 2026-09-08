#!/bin/bash
#SBATCH --job-name=sip_armE_CA_eval
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_integrated/slurm/orgslot/logs/WORD070_ArmE/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_integrated/slurm/orgslot/logs/WORD070_ArmE/%x_%j.err
set -euo pipefail
: "${EVAL_WORKDIR:?export the matching frozen training worktree}"
export EVAL_WORKDIR
export RUN_TAG=OrgSlot_WORD07072_ROI20_ArmE_CrossAttn_BF16
export EXPECTED_HEAD_TYPE=arm_e_multiscale_query
: "${TRAIN_JOB_ID:?export the matching successful training job ID}"
cd "${EVAL_WORKDIR}"
export PYTHONNOUSERSITE=1
export PATH=/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=/gpfs/work/aac/bolinren19/.conda/envs/abdpet/lib:${LD_LIBRARY_PATH:-}
PYTHON=/gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python
CHECKPOINT="${EVAL_WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_TAG}_${TRAIN_JOB_ID}/checkpoint-802.pth"
# Fixed selection based on GT, not prediction quality. No fabricated trained heatmaps.
${PYTHON} -m tools.export_query_cross_attention --checkpoint "${CHECKPOINT}" \
 --data-csv /gpfs/work/aac/bolinren19/2026-07/DATA/processed/OWT_Common8_07072_NATIVE/WORD/csv/WORD_Test_2D_native07072.csv \
 --output-dir "${EVAL_WORKDIR}/artifacts/query_cross_attention/${TRAIN_JOB_ID}_${SLURM_JOB_ID}" \
 --class-ids 4,5,6 --samples-per-organ 2
bash slurm/orgslot/eval/orgslot_word07072_unified_eval_bolin_sip.sbatch
