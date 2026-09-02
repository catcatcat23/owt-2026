#!/bin/bash
#SBATCH --job-name=sip_os070_armE
#SBATCH --partition=gpua800
#SBATCH --account=angelosstefanidis
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
#SBATCH --output=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiscale_pixel/slurm/orgslot/logs/WORD070_ArmE/%x_%j.out
#SBATCH --error=/gpfs/work/aac/bolinren19/OD_OWT/.worktrees/orgslot_querymask_multiscale_pixel/slurm/orgslot/logs/WORD070_ArmE/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/bolinren19
WORKDIR=${ROOT}/OD_OWT/.worktrees/orgslot_querymask_multiscale_pixel
RUN_NAME=OrgSlot_WORD07072_ROI20_ArmE_MultiscalePixel_SmallOrgan_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/.conda/envs/abdpet/bin/python
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
export N_GPU=4 MICRO_BATCH=8 ACCUM_ITER=6 TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800 WARMUP_UPDATES=5940
export BASE_LR=1e-4 WEIGHT_DECAY=0.05 WORKERS=6 SAVE_FREQ=100
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/.conda/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/.conda/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

test "$((N_GPU * MICRO_BATCH * ACCUM_ITER))" -eq 192
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -r "${OUTPUT_DIR}/checkpoint-802.pth"
${PYTHON} -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-802.pth', map_location='cpu'); a=c['args']; assert int(c['epoch']) == 802; assert a.slot_head_type == 'arm_e_multiscale_query'; assert list(a.expected_spacing) == [0.7, 0.7, 2.0]; assert a.seg_supervision == 'retained'; assert a.seg_loss_type == 'small_organ'; assert float(a.lambda_seg) == 0.01; assert float(a.lambda_bg_seg) == 0; assert float(a.organ_roi_probability) == 0.2; assert any(k.startswith('pixel_query_decoder.spatial_stem.') for k in c['model']); print('formal Arm E checkpoint validated')"
