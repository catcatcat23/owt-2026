
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/ImageNet/imagenet/

## BTCV
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 12 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med-SAM2D/ProMISe_Dataset/Datasets/synapseCT/Training/2D_all_5slice/training_full_path.csv #> tmp.txt & 

## AbdAtlas
# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.5 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 9 --arch_version v1 --token_factor 20 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv > tmp.txt & 

# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_large_patch16 --norm_pix_loss --mask_ratio 0.5 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 9 --arch_version v1 --token_factor 40 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv > tmp.txt & 

####################################################
#### ARCH_Ver 
## v1 OWC with cls token in decoder; 
## ~~v11 OWC no cls token in decoder~~;
## v2 OWC with cls token in decoder + VQGAN decoder;
####################################################
#### TRAIN_Ver 
## v0 mask organ tokens; 
## v1 complement/impaint;
####################################################
## no norm_pix_loss, with augmentation, with pos embedding
############################################################################################################
cd /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='0,1,2,3,4,5,6,7' # '0,1,2,3' '4,5,6,7'
N_GPU=8
CUR_PATH=AbdAtlas1000
MODEL_NAME=mae_vit_base_patch16
BASE_LR=1.5e-4
CLASSES=9
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
BATCH_SIZE=16 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB)
EPOCH=400 ## v1: 800-40; v2: 400-20
WARMUP=20
ARCH_Ver=v2 ## v1, v2; -VQv0; -DNv0
TRAIN_Ver=v0 ## v0
####################################################
TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv
RESULT_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 50 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt & 

ARCH_Ver=v1
TRAIN_Ver=v0 ## v0 generate; v1 complement
CUDA_VISIBLE_DEVICES=4 python vis_mae_token.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor 20 --training_version ${TRAIN_Ver} --checkpoint /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/AbdAtlas1000/Token_mae_vit_base_patch16_1.5e-4_9_224_20_${ARCH_Ver}_${TRAIN_Ver}/checkpoint-799.pth --select_cls '5'
#'/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/bak/bak3_output_dir_withbg/checkpoint-560.pth'










## vis test
CUDA_VISIBLE_DEVICES=4 python vis_mae_token.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version v1 --token_factor 20 #--data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv


## test (ONCE)
# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 12 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med-SAM2D/ProMISe_Dataset/Datasets/synapseCT/Training/2D_all_5slice/training_full_path.csv 