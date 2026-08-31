#!/bin/bash
#SBATCH --job-name=xec_os070_maeE
#SBATCH --partition=gpua800,gpua8001t
#SBATCH --account=sifansong
#SBATCH --qos=4gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a800:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=4-00:00:00
#SBATCH --output=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_AutoPETMAE/%x_%j.out
#SBATCH --error=/gpfs/work/aac/sifansong/slurm/logs/OrganSlot/WORD070_AutoPETMAE/%x_%j.err

set -euo pipefail
ROOT=/gpfs/work/aac/sifansong
WORKDIR=${ROOT}/worktrees/orgslot_autopet_mae
RUN_TAG=OrgSlot_WORD07072_ROI20_RetainedMultiConv_SmallOrgan_L001_AutoPETMAE_EncoderOnly
RUN_NAME=${RUN_TAG}_${SLURM_JOB_ID}
OUTPUT_DIR=${WORKDIR}/Results/OrganSlotBank/Common8/WORD_2D/${RUN_NAME}
MAE_INIT_CHECKPOINT=${ROOT}/Results/AutoPET_MAE/MAE_LA6D8_224_M75_U100K_seed0/checkpoint-final.pth

cd "${WORKDIR}"
export PYTHON=${ROOT}/envs/abdpet/bin/python
export DATA_ROOT=${ROOT}/data
export PROCESSED_ROOT=${ROOT}/data/OWT_Common8_07072_NATIVE/WORD
export MANIFEST_TAG=native07072
export EXPECTED_SPACING="0.7 0.7 2"
export ROI_INDEX=${PROCESSED_ROOT}/metadata/small_organ_roi_index_verified.json
export LPIPS_STATE=${ROOT}/data/pretrained/owt_lpips_vgg16.pth
export FUSION_MODE=linear_sqrt FUSION_REFERENCE_COUNT=9
export LAMBDA_SEG=0.01 LAMBDA_BG_SEG=0 SEG_SUPERVISION=retained
export SEG_LOSS_TYPE=small_organ FOCAL_ALPHA=0.75 FOCAL_GAMMA=2.0
export TVERSKY_ALPHA_FP=0.3 TVERSKY_BETA_FN=0.7 TVERSKY_EPS=1e-6
export BALANCED_FOCAL_WEIGHT=0.5 HARD_NEGATIVE_RATIO=0.02
export NEGATIVE_SLICE_WEIGHT=0.1
export SLOT_HEAD_TYPE=multiscale_conv SLOT_HEAD_CHANNELS=128
export ORGAN_ROI_PROBABILITY=0.2 TGR_MODE=legacy_batch
export MAE_INIT_CHECKPOINT MAE_INIT_SCOPE=encoder
export RUN_NAME OUTPUT_DIR
export N_GPU=4 MICRO_BATCH=8 ACCUM_ITER=6 TARGET_EFFECTIVE_BATCH=192
export MAX_UPDATES=118800 WARMUP_UPDATES=5940 BASE_LR=1e-4 WEIGHT_DECAY=0.05
export WORKERS=6 SAVE_FREQ=100
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export PATH=${ROOT}/envs/abdpet/bin:${PATH}
export LD_LIBRARY_PATH=${ROOT}/envs/abdpet/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

test -s "${MAE_INIT_CHECKPOINT}"
"${PYTHON}" tools/validate_word070_dataset_identity.py --processed_root "${PROCESSED_ROOT}"
test -s "${LPIPS_STATE}"
echo "a99ef0ef436727d809148f5e9cbc09c1f0218079020918b0b1cc5167493c4287  ${LPIPS_STATE}" | sha256sum -c -
test "${ORGAN_ROI_PROBABILITY}" = "0.2"
test "${EXPECTED_SPACING}" = "0.7 0.7 2"
test "$((N_GPU * MICRO_BATCH * ACCUM_ITER))" -eq 192
nvidia-smi -L
"${PYTHON}" -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 4"
bash scripts/orgslot/run_word_112_448_a100.sh roi20

test -r "${OUTPUT_DIR}/checkpoint-802.pth"
test -r "${OUTPUT_DIR}/mae_initial_checkpoint_load.json"
"${PYTHON}" -c "import json,torch; c=torch.load('${OUTPUT_DIR}/checkpoint-802.pth', map_location='cpu'); a=c['args']; r=json.load(open('${OUTPUT_DIR}/mae_initial_checkpoint_load.json')); assert int(c['epoch'])==802; assert list(a.expected_spacing)==[0.7,0.7,2.0]; assert a.slot_head_type=='multiscale_conv'; assert a.seg_loss_type=='small_organ'; assert float(a.lambda_seg)==0.01; assert a.mae_init_scope=='encoder'; assert r['scope']=='encoder'; assert r['target_input_size']==448; print('WORD AutoPET-MAE transfer checkpoint validated')"
