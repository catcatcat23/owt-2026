#!/bin/bash
#SBATCH --job-name=sip_2dD_spatialPE
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiquery/slurm/orgslot/logs/WORD070_MultiQueryMask/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiquery/slurm/orgslot/logs/WORD070_MultiQueryMask/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/bolinren19
WORKDIR=${ROOT}/OD_OWT/.worktrees/orgslot_pixel_pe
export PIXEL_PE=spatial
export AMP_DTYPE=bf16 CLIP_GRAD=1.0 FINITE_CHECK_INTERVAL=1
RUN_TAG=OrgSlot_WORD07072_ROI20_MultiQueryDot_SmallOrgan_2D_SpatialPE
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/.conda/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/2026-07/DATA
export PROCESSED_ROOT=${DATA_ROOT}/processed/OWT_Common8_07072_NATIVE/WORD
export MANIFEST_TAG=native07072
export EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index.json
export LPIPS_STATE=${ROOT}/OD_OWT_orgslot/Results/OrganSlotBank/_assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=small_organ FOCAL_ALPHA=0.75 FOCAL_GAMMA=2.0
export TVERSKY_ALPHA_FP=0.3 TVERSKY_BETA_FN=0.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=0.5 HARD_NEGATIVE_RATIO=0.02
export NEGATIVE_SLICE_WEIGHT=0.1
export SLOT_HEAD_TYPE=multi_query_dot SLOT_HEAD_CHANNELS=128
export ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export RUN_NAME OUTPUT_DIR
export N_GPU=4 MICRO_BATCH=8 ACCUM_ITER=6 TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 WEIGHT_DECAY=0.05
export WORKERS=6 SAVE_FREQ=25
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/.conda/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

test "${ORGAN_ROI_PROBABILITY}" = "0.2"
test "${EXPECTED_SPACING}" = "0.7 0.7 2"
test "$((N_GPU * MICRO_BATCH * ACCUM_ITER))" -eq 192
nvidia-smi -L
${PYTHON} -c "import os, torch; n=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); assert torch.cuda.is_available() and n == 4"
# Validate the real BF16 DDP pipeline in this allocation, then start fresh.
${PYTHON} -c "import torch; assert torch.cuda.is_bf16_supported()"
(
  export OUTPUT_DIR="${OUTPUT_DIR}_preflight" RUN_NAME="${RUN_NAME}_preflight"
  export MAX_UPDATES=2 WARMUP_UPDATES=0 SAVE_FREQ=1
  bash scripts/orgslot/run_word_112_448_a100.sh roi20
  ${PYTHON} tools/validate_orgslot_jointseg_smoke.py \
    --run-dir "${OUTPUT_DIR}" --expected-seg-loss small_organ \
    --expected-seg-supervision retained --expected-slot-head-type multi_query_dot \
    --expected-background-weight 0 --expected-lambda-seg 0.01 --expected-spacing 0.7 0.7 2
)
bash scripts/orgslot/run_word_112_448_a100.sh roi20

test -r "${OUTPUT_DIR}/checkpoint-802.pth"
${PYTHON} -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-802.pth', map_location='cpu'); a=c['args']; s=c['model']; assert int(c['epoch']) == 802; assert list(getattr(a, 'expected_spacing')) == [0.7, 0.7, 2.0]; assert getattr(a, 'seg_supervision') == 'retained'; assert getattr(a, 'seg_loss_type') == 'small_organ'; assert getattr(a, 'slot_head_type') == 'multi_query_dot'; assert float(getattr(a, 'lambda_seg')) == 0.01; assert float(getattr(a, 'lambda_bg_seg')) == 0; assert float(getattr(a, 'organ_roi_probability')) == 0.2; assert any(k.startswith('pixel_query_decoder.') for k in s); print('formal multi-query checkpoint validated')"
