#!/bin/bash
#SBATCH --job-name=xec_os070_3dC
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_ArmC3D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_ArmC3D/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/orgslot_querymask_3d
RUN_TAG=OrgSlot_WORD07072_ROI20_RetainedQueryDot3D_FixFr4_SmallOrgan_L0.01_BF16_Clip1
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_3D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/data
export PROCESSED_ROOT=${DATA_ROOT}/OWT_Common8_07072_NATIVE/WORD
export MANIFEST_TAG=native07072
export EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_verified.json
export LPIPS_STATE=${DATA_ROOT}/assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=small_organ FOCAL_ALPHA=0.75 FOCAL_GAMMA=2.0
export TVERSKY_ALPHA_FP=0.3 TVERSKY_BETA_FN=0.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=0.5 HARD_NEGATIVE_RATIO=0.02
export NEGATIVE_SLICE_WEIGHT=0.1
export AMP_DTYPE=bf16 CLIP_GRAD=1.0 FINITE_CHECK_INTERVAL=25
export SLOT_HEAD_TYPE=query_dot SLOT_HEAD_CHANNELS=128
export DIMENSION=3D FIX_FRAME=4 TEMP_STRIDE=1
export ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export RUN_NAME OUTPUT_DIR

# A 4-slice slab is one optimizer sample. Forty-eight slabs per update expose
# approximately 192 slices, matching Arm B's 2D slice-equivalent batch.
export N_GPU=4 MICRO_BATCH=2 ACCUM_ITER=6 TARGET_EFFECTIVE_BATCH=48
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 WEIGHT_DECAY=0.05
export ACTUAL_LR=7.5e-5
export WORKERS=6 SAVE_FREQ=10
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/envs/abdpet/bin:${PATH}
export PYTHONNOUSERSITE=1

test "${ORGAN_ROI_PROBABILITY}" = "0.2"
${PYTHON} tools/validate_word070_dataset_identity.py --processed_root "${PROCESSED_ROOT}"
test "${EXPECTED_SPACING}" = "0.7 0.7 2"
test "${DIMENSION}" = "3D"
test "${FIX_FRAME}" = "4"
test "$((N_GPU * MICRO_BATCH * ACCUM_ITER))" -eq 48
test "$((N_GPU * MICRO_BATCH * ACCUM_ITER * FIX_FRAME))" -eq 192
test "${ACTUAL_LR}" = "7.5e-5"
test "${AMP_DTYPE}" = "bf16"
test "${SLOT_HEAD_TYPE}" = "query_dot"
nvidia-smi -L
${PYTHON} -c "import os, torch; n=torch.cuda.device_count(); print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); assert torch.cuda.is_available() and n == 4"
bash scripts/orgslot/run_word_112_448_a100.sh roi20

test -r "${OUTPUT_DIR}/checkpoint-199.pth"
${PYTHON} -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-199.pth', map_location='cpu'); a=c['args']; assert int(c['epoch']) == 199; assert getattr(a, 'dimension') == '3D'; assert int(getattr(a, 'fix_frame')) == 4; assert list(getattr(a, 'expected_spacing')) == [0.7, 0.7, 2.0]; assert getattr(a, 'seg_loss_type') == 'small_organ'; assert getattr(a, 'slot_head_type') == 'query_dot'; assert float(getattr(a, 'lambda_seg')) == 0.01; assert float(getattr(a, 'organ_roi_probability')) == 0.2; assert getattr(a, 'amp_dtype') == 'bf16'; assert float(getattr(a, 'clip_grad')) == 1.0; print('formal 3D Arm C checkpoint validated')"
