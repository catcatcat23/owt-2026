#!/bin/bash
#SBATCH --job-name=sip_os070_armE_2g
#SBATCH --partition=gpua800
#SBATCH --account=sifansong
#SBATCH --qos=4a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=112G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiscale_pixel/slurm/orgslot/logs/WORD070_ArmE/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiscale_pixel/slurm/orgslot/logs/WORD070_ArmE/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/bolinren19
WORKDIR=${ROOT}/OD_OWT/.worktrees/orgslot_querymask_multiscale_pixel
RUN_NAME=OrgSlot_WORD07072_ROI20_ArmE_SmallOrgan_Smoke2G_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/.conda/envs/abdpet/bin/python
export PATH=${ROOT}/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/.conda/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))

${PYTHON} -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  tools/smoke_arm_e_ddp.py

export DATA_ROOT=${ROOT}/2026-07/DATA
export PROCESSED_ROOT=${DATA_ROOT}/processed/OWT_Common8_07072_NATIVE/WORD
export MANIFEST_TAG=native07072 EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index.json
export LPIPS_STATE=${ROOT}/OD_OWT_orgslot/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=small_organ FOCAL_ALPHA=0.75 FOCAL_GAMMA=2.0
export TVERSKY_ALPHA_FP=0.3 TVERSKY_BETA_FN=0.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=0.5 HARD_NEGATIVE_RATIO=0.02
export NEGATIVE_SLICE_WEIGHT=0.1
export SLOT_HEAD_TYPE=arm_e_multiscale_query SLOT_HEAD_CHANNELS=128
export ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export RUN_NAME OUTPUT_DIR
export N_GPU=2 MICRO_BATCH=4 ACCUM_ITER=1 TARGET_EFFECTIVE_BATCH=8
export MAX_UPDATES=4 WARMUP_UPDATES=0 WORKERS=6 SAVE_FREQ=1

bash scripts/orgslot/run_word_112_448_a100.sh roi20
${PYTHON} tools/validate_orgslot_jointseg_smoke.py \
  --run-dir "${OUTPUT_DIR}" \
  --expected-seg-loss small_organ \
  --expected-seg-supervision retained \
  --expected-slot-head-type arm_e_multiscale_query \
  --expected-background-weight 0 \
  --expected-lambda-seg 0.01 \
  --expected-spacing 0.7 0.7 2
grep -q "ARM_E_DDP_DIFFERENT_ORGANS_OK" "slurm/orgslot/logs/WORD070_ArmE/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
