#!/bin/bash
#SBATCH --job-name=xec_os070_3dC_smk
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_ArmC3D/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_ArmC3D/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/orgslot_querymask_3d
RUN_NAME=Smoke_WORD07072_ROI_QueryDot3D_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/_smoke/WORD_3D/${RUN_NAME}
cd "${WORKDIR}"

export PYTHON=${ROOT}/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/data
export PROCESSED_ROOT=${DATA_ROOT}/OWT_Common8_07072_NATIVE/WORD
export MANIFEST_TAG=native07072 EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_verified.json
export LPIPS_STATE=${DATA_ROOT}/assets/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=small_organ FOCAL_ALPHA=0.75 FOCAL_GAMMA=2.0
export TVERSKY_ALPHA_FP=0.3 TVERSKY_BETA_FN=0.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=0.5 HARD_NEGATIVE_RATIO=0.02 NEGATIVE_SLICE_WEIGHT=0.1
export AMP_DTYPE=bf16 CLIP_GRAD=1.0 FINITE_CHECK_INTERVAL=1
export SLOT_HEAD_TYPE=query_dot SLOT_HEAD_CHANNELS=128
export DIMENSION=3D FIX_FRAME=4 TEMP_STRIDE=1 TGR_MODE=legacy_batch
# Force the ROI path in this diagnostic run. The formal experiment remains 0.2.
export ORGAN_ROI_PROBABILITY=1.0
export RUN_NAME OUTPUT_DIR
export N_GPU=1 MICRO_BATCH=1 ACCUM_ITER=1 TARGET_EFFECTIVE_BATCH=1
export MAX_UPDATES=2 WARMUP_UPDATES=0 ACTUAL_LR=7.5e-5
export MAX_TRAIN_SAMPLES=8 MAX_VAL_SAMPLES=4 WORKERS=4 SAVE_FREQ=1
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/envs/abdpet/bin:${PATH} PYTHONNOUSERSITE=1

${PYTHON} tools/validate_word070_dataset_identity.py --processed_root "${PROCESSED_ROOT}"
nvidia-smi -L
bash scripts/orgslot/run_word_112_448_a100.sh roi20
${PYTHON} tools/validate_orgslot_jointseg_smoke.py \
  --run-dir "${OUTPUT_DIR}" \
  --expected-seg-loss small_organ \
  --expected-seg-supervision retained \
  --expected-slot-head-type query_dot \
  --expected-background-weight 0 \
  --expected-lambda-seg 0.01 \
  --expected-spacing 0.7 0.7 2
${PYTHON} -c "import torch; c=torch.load('${OUTPUT_DIR}/checkpoint-0.pth', map_location='cpu'); assert c['args'].dimension == '3D'; assert c['args'].slot_head_type == 'query_dot'; print('3D Arm C real-data smoke validated')"
