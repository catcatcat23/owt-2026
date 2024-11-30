
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
sbatch -J model_train2::proj=IRB2021P002249, -p defq -n 1 --cpus-per-task=128 --gres=gpu:8 -t 3-12:00:00 -w emimbmgpu40-01 tmp/job_tmp1.sh
sbatch -J model_train2::proj=IRB2021P002249, -p defq -n 1 --cpus-per-task=1 --gres=gpu:1 -t 1-00:00:00 -w emimbmgpu40-04 tmp/vis_tmp1.sh
############################################################################################################
cd /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/
PORT_NUM=25664
GPU_num='0,1,2,3,4,5,6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=8
BASE_LR=1e-4 #4e-4 #1.5e-5 #8e-5 ## 1e-4 1.5e-4 5e-5 4.5e-6
INPUT_SIZE=224
TOKEN_Fac=20 #200 #20 for 2d 200 for 3d
####################################################
CUR_PATH=Abdomen1k_3D ## AbdAtlas1000, Abdomen1k
CLASSES=5 ## 9 for AbdAtlas1000; 5 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=16 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB); 1 for 3D (temp); 48 for -LA -Fixfr16; 12 for -Fixfr16 (w/o -LA)
EPOCH=1200 ## v1: 800-40; v2: 400-20
WARMUP=60 ## v1: 800-40; v2: 400-20
ARCH_Ver=v11
####     v1 (Trans Decoder), v2 (VQGAN Decoder), v3 (VQGAN Encoder to 1/16hw before two transformers);
####         v11; v31: without intermediate cls tokens
####         -cls4 (用于skip的 slice/global information) (currently, only for v3)
####     -VQv0/v01/v1_nt512; (v01比v0用的non-linear; v1用的NormEMA QT)
####         -LIBv0 (是否一个organ一个codebook); -DTv0;
TRAIN_Ver=v01-3D-Fixfr16-TS1 ## v0 (generation) (v01 generation without mask tokens); v1 (inpainting); v01-3D-Fixfr4-TS1
####     -3D -Fixfr16
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
# TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv
TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr16_Final112.csv
RESULT_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt & 
## --mask_ratio 0.95 currently only for ARCH_Ver=v1-VQv0_nt128 BASE_LR=8e-5

ARCH_Ver=v1
TRAIN_Ver=v0 ## v0 generate; v1 complement
TOKEN_Fac=10
LOSS_Ver=L2
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
CUDA_VISIBLE_DEVICES=7 python vis_mae_token.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --checkpoint /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/AbdAtlas1000/${PROCESS_NAME}/checkpoint-799.pth --select_cls '1,2,3,4,5,6,7,8,9'
#'/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/bak/bak3_output_dir_withbg/checkpoint-560.pth'










## vis test
CUDA_VISIBLE_DEVICES=4 python vis_mae_token.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version v1 --token_factor 20 #--data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv


## test (ONCE)
# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 12 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med-SAM2D/ProMISe_Dataset/Datasets/synapseCT/Training/2D_all_5slice/training_full_path.csv 