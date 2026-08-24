#!/bin/bash
#SBATCH --job-name=sip_focal_r500_smk
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_head_ab/slurm/orgslot/logs/WORD448_FocalResume/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_head_ab/slurm/orgslot/logs/WORD448_FocalResume/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/bolinren19
WORKDIR=${ROOT}/OD_OWT/.worktrees/orgslot_head_ab
SOURCE_RUN=OrgSlot_WORD448_ROI20_RetainedMultiConv_Focal_L0.1_119641
RESUME_CHECKPOINT=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${SOURCE_RUN}/checkpoint-500.pth
RUN_TAG=OrgSlot_WORD448_ROI20_RetainedMultiConv_Focal_L0.1_Resume500_BolinSmoke
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/.conda/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/2026-07/DATA
export PROCESSED_ROOT=${DATA_ROOT}/processed/OWT_Common8_112_NATIVE/WORD
export MANIFEST_TAG=native112
export EXPECTED_SPACING="1 1 2"
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index.json
export LPIPS_STATE=${ROOT}/OD_OWT_orgslot/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.1 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=focal FOCAL_ALPHA=0.75 FOCAL_GAMMA=2.0
export SLOT_HEAD_TYPE=multiscale_conv SLOT_HEAD_CHANNELS=128
export ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export RUN_NAME OUTPUT_DIR RESUME_CHECKPOINT
export N_GPU=2 MICRO_BATCH=8 ACCUM_ITER=12 TARGET_EFFECTIVE_BATCH=192
# Full-data updates_per_epoch=148. Epoch 501 begins at update 74148; stop after 2 updates.
export MAX_UPDATES=74150 WARMUP_UPDATES=5940 MAX_STEPS_PER_EPOCH=24
export BASE_LR=1e-4 WEIGHT_DECAY=0.05 WORKERS=10 SAVE_FREQ=1
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/.conda/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

test "$(sha256sum "${RESUME_CHECKPOINT}" | cut -d' ' -f1)" = "aca81047809978f6a1f818c468202485895a48ab20f2750182e54124fdf02dab"
test "$((N_GPU * MICRO_BATCH * ACCUM_ITER))" -eq 192
nvidia-smi -L
${PYTHON} -c "import os, torch; n=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); assert torch.cuda.is_available() and n == 2"
bash scripts/orgslot/run_word_112_448_a100.sh roi20

test -r "${OUTPUT_DIR}/checkpoint-501.pth"
${PYTHON} tools/validate_orgslot_jointseg_smoke.py \
  --run-dir "${OUTPUT_DIR}" \
  --expected-seg-loss focal \
  --expected-seg-supervision retained \
  --expected-slot-head-type multiscale_conv \
  --expected-background-weight 0 \
  --expected-lambda-seg 0.1 \
  --expected-focal-alpha 0.75 \
  --expected-focal-gamma 2.0 \
  --expected-spacing 1 1 2
${PYTHON} -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-501.pth', map_location='cpu'); assert int(c['epoch']) == 501; assert int(c['args'].start_epoch) == 501; print('resume smoke checkpoint validated')"
