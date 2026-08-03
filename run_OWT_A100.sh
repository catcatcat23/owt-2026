#######################
#### Main Training ####
#######################
## 3D datasets (Abdomen1k, AutoPet, RAOS/Delay, RAOS/PreArtery)
## 4-slice training for medical image datasets ## 
####################################################
SERVER_PATH=/mnt/DATA-4/sifan2 #/raid/camca/ss3112 ## H100_1
cd ${SERVER_PATH}/Scripts/OWT/
####################################################
PORT_NUM=25666
GPU_num='1,3'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_3D
CLASSES=4
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA  ## -LA (linear attention)
BATCH_SIZE=64 
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr4-TS1 
LOSS_Ver=L2-LPIPS ## L2; -LPIPS
####################################################
TRAIN_CSV_PATH=/gpfs/work/aac/bolinren19/2026-07/Training_Fixfr4_A100_Final112.csv
RESULT_PATH=${SERVER_PATH}/Scripts/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Training/image/
TRAIN_LABL_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Training/mask/
TEST_DATA_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Test/image/
TEST_LABL_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Test/mask/

## 1-slice training for medical image datasets ## 
####################################################
SERVER_PATH=/mnt/DATA-4/sifan2 ## H100_1
cd ${SERVER_PATH}/Scripts/OWT/
####################################################
PORT_NUM=25666
GPU_num='1,3'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_2D 
CLASSES=4 
####################################################
MODEL_NAME=mae_vit_base_patch16-LA  ## -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; -LPIPS
####################################################
TRAIN_CSV_PATH=/gpfs/work/aac/bolinren19/2026-07/Training_Fixfr4_A100_Final112.csv
RESULT_PATH=${SERVER_PATH}/Scripts/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Training/image/
TRAIN_LABL_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Training/mask/
TEST_DATA_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Test/image/
TEST_LABL_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Test/mask/

#########################
#### Step1: Training ####
#########################
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt