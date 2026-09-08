#!/bin/bash
#SBATCH --job-name=xec_armE_MAE
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
set -euo pipefail
: "${EXPERIMENT_WORKDIR:?}"
: "${MAE_INIT_SCOPE:?}"
case "${MAE_INIT_SCOPE}" in encoder|encoder_decoder) ;; *) exit 2 ;; esac
ROOT=/gpfs/work/aac/sifansong
cd "${EXPERIMENT_WORKDIR}"
export PYTHON=${ROOT}/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/data
export PROCESSED_ROOT=${ROOT}/data/OWT_Common8_07072_NATIVE/WORD
export MANIFEST_TAG=native07072 EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_verified.json
export LPIPS_STATE=${ROOT}/data/pretrained/owt_lpips_vgg16.pth
export MAE_INIT_CHECKPOINT=${ROOT}/Results/AutoPET_MAE/MAE_LA6D8_224_M75_U100K_seed0/checkpoint-final.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=small_organ FOCAL_ALPHA=0.75 FOCAL_GAMMA=2.0
export TVERSKY_ALPHA_FP=0.3 TVERSKY_BETA_FN=0.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=0.5 HARD_NEGATIVE_RATIO=0.02 NEGATIVE_SLICE_WEIGHT=0.1
export SLOT_HEAD_TYPE=arm_e_multiscale_query SLOT_HEAD_CHANNELS=128
export ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export RUN_NAME=OrgSlot_WORD07072_ROI20_ArmE_MAE_${MAE_INIT_SCOPE}_${SLURM_JOB_ID}
export OUTPUT_DIR=${EXPERIMENT_WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
export N_GPU=4 MICRO_BATCH=8 ACCUM_ITER=6 TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 WEIGHT_DECAY=0.05
export WORKERS=6 SAVE_FREQ=100 POSITIVE_ROI_LOSS_WEIGHT=0 RESUME_CHECKPOINT=""
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/envs/abdpet/lib:/gpfs/spack/opt/spack/linux-icelake/mesa-25.0.5-zygiclacf7ay3uqqvqizl3lsvuzkk4od/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1
echo "0b3571a3e79095686a0b12649735a298aaeb8a0b3bb93a47f3484439ac4a1ec6  ${MAE_INIT_CHECKPOINT}" | sha256sum -c -
"${PYTHON}" tools/validate_word070_dataset_identity.py --processed_root "${PROCESSED_ROOT}"
# Frozen 81adba2 engine: original Arm E FP16 + GradScaler, no new clipping or PE.
bash scripts/orgslot/run_word_112_448_a100.sh roi20
test -s "${OUTPUT_DIR}/checkpoint-802.pth"
test -s "${OUTPUT_DIR}/mae_initial_checkpoint_load.json"
