#!/bin/sh

#SBATCH --output=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/tmp/%j.out
#SBATCH --error=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/tmp/%j.out

############################################################################################################
cd /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/
PORT_NUM=25634
GPU_num='0,1,2,3,4,5,6,7' # '0,1,2,3' '4,5,6,7'
N_GPU=8
CUR_PATH=AbdAtlas1000
MODEL_NAME=mae_vit_base_patch16
BASE_LR=1e-4 #8e-5 ## 1.5e-4 5e-5 4.5e-6
CLASSES=9
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
BATCH_SIZE=64 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB)
EPOCH=800 ## v1: 800-40; v2: 400-20
WARMUP=40 ## v1: 800-40; v2: 400-20
ARCH_Ver=v11
####     v1 (Trans Decoder), v2 (VQGAN Decoder), v3 (VQGAN Encoder to 1/16hw before two transformers);
####         -cls4 (用于skip的 slice/global information) (currently, only for v3)
####     -VQv0/v01/v1_nt512; (v01比v0用的non-linear; v1用的NormEMA QT)
####         -LIBv0 (是否一个organ一个codebook); -DTv0;
TRAIN_Ver=v0 ## v0 (generation); v1 (inpainting)
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv
RESULT_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt 
## --mask_ratio 0.95 currently only for ARCH_Ver=v1-VQv0_nt128 BASE_LR=8e-5

