###########################################################
## Abdomen1k_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='0,5,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=3
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/Abdomen1k_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/
###########################################################
## Abdomen1k_2D final 1 slice (2D-MAE-NO LPIPS) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='2,3' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/Abdomen1k_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2 ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
# ###########################################################
# ## Abdomen1k_3D final 1 slice (3D-MAE) (not finished)
# ###########################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25695
# GPU_num='0,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=4
# BASE_LR=1e-4
# INPUT_SIZE=224
# TOKEN_Fac=1
# MASK_RATIO=0.75
# ####################################################
# CUR_PATH=MAE/Abdomen1k_3D ## AbdAtlas1000, Abdomen1k
# CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
# ####################################################
# MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=64 
# EPOCH=1200
# WARMUP=60
# ARCH_Ver=v0
# TRAIN_Ver=v0-3D-Fixfr4-TS1
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr4_H100_2_Final112.csv
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
# ###########################################################


###########################################################
## AutoPet_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='2,3' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
###########################################################
## AutoPet_2D final 1 slice (2D-MAE-NO LPIPS) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='2,3' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2 ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################


###########################################################
## Delay_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25624
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/SyntheticMRI_Delay_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/mask/
###########################################################
## Delay_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25635
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/SyntheticMRI_Delay_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2 ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################


###########################################################
## PreArtery_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25618
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/SyntheticMRI_PreArtery_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/mask/
###########################################################
## PreArtery_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25637
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/SyntheticMRI_PreArtery_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2 ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################


#########################
#### Compare Step1: Pretrain/mae ####
#########################
# CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio ${MASK_RATIO} --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt

########################################################
#### Compare Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=3
ckpt=1199
THRE=(0.02) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=(',')
MASK=(0.0 0.25 0.75) ## 0.2 0.4 0.6 0.8
for cls_i in {0..0..1}
do
for thre in {0..0..1}
do
for mask in {0..2..1}
do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen_maelike.py --model ${MODEL_NAME} --mask_ratio ${MASK[${mask}]} --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}_maskrm${MASK[${mask}]}.txt
done
done
done


#### ICCV Rebuttal vae/vqgan ####
###########################################################
## Abdomen1k_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='0,1,2,3,4,5,6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=8
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.0
####################################################
CUR_PATH=VAE/Abdomen1k_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=vae_cnn_base ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v012 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Final112.csv
RESULT_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
TRAIN_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
TEST_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
TEST_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/
###########################################################
## Abdomen1k_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='1,2,3,4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=5
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.0
####################################################
CUR_PATH=VAE/Abdomen1k_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=vae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v01 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/


###########################################################
## AutoPet_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/
PORT_NUM=25602
GPU_num='0,1,2,3,4,5,6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=8
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.0
####################################################
CUR_PATH=VAE/AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=vae_cnn_base ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v012 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Final112.csv
RESULT_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
# ###########################################################
# ## AutoPet_2D final 1 slice (2D-MAE) (finished)
# ###########################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25694
# GPU_num='4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=3
# BASE_LR=1e-4
# INPUT_SIZE=224
# TOKEN_Fac=1
# MASK_RATIO=0.0
# ####################################################
# CUR_PATH=VAE/AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
# CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
# ####################################################
# MODEL_NAME=vae_cnn_base ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=96 
# EPOCH=1200
# WARMUP=60
# ARCH_Ver=v012 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
# TRAIN_Ver=v0
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
# ###########################################################
# TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
# TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
# TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
# TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/

###########################################################
## PreArtery_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/
PORT_NUM=25612
GPU_num='0,1,2,3,4,5,6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=8
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.0
####################################################
CUR_PATH=VAE/SyntheticMRI_Delay_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=vae_cnn_base ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v012 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_Final112.csv
RESULT_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/image/
TRAIN_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/mask/
TEST_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/image/
TEST_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/mask/

###########################################################
## PreArtery_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25622
GPU_num='0,1' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.0
####################################################
CUR_PATH=VAE/SyntheticMRI_PreArtery_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=vae_cnn_base ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v012 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
TRAIN_Ver=v0
LOSS_Ver=L2 ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/mask/
###########################################################
## PreArtery_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25612
GPU_num='2,3' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.0
####################################################
CUR_PATH=VAE/SyntheticMRI_PreArtery_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=vae_cnn_base ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v012 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
###########################################################
## PreArtery_2D final 1 slice (2D-MAE) (finished)
###########################################################
cd /mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/
PORT_NUM=25612
GPU_num='0,1,2,3,4,5,6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=8
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.0
####################################################
CUR_PATH=VAE/SyntheticMRI_PreArtery_2D_rep2 ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=vae_cnn_base ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v012 ## v01 vae-vit, v011 vae_cnn, v012 vae_vallina, v02 vqgan, v021 vqgan cnn
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_Final112.csv
RESULT_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/image/
TRAIN_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/mask/
TEST_DATA_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/image/
TEST_LABL_PATH=/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/mask/

## for vae/vqgan
#########################
#### Compare Step1: Pretrain/vae/vqgan ####
#########################
# CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio ${MASK_RATIO} --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 400 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt
########################################################
#### Compare Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=1
ckpt=1199
THRE=(0.02) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=(',')
MASK=(0.0 0.25 0.75) ## 0.2 0.4 0.6 0.8
for cls_i in {0..0..1}
do
for thre in {0..0..1}
do
for mask in {0..0..1}
do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen_maelike.py --model ${MODEL_NAME} --mask_ratio ${MASK[${mask}]} --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}_maskrm${MASK[${mask}]}.txt
done
done
done

########################################################
Pure2D######################################################## ## (current only face)
########################################################
####################################################################
## Face final 1 slice (2D-Trans) (finished)
####################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25699
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/CelebAMaskHQ_2D ## AbdAtlas1000, Abdomen1k
CLASSES=5 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
##################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Test/mask/
####################################################################
## Face final 1 slice (2D-Trans) (finished)
####################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25699
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=1
MASK_RATIO=0.75
####################################################
CUR_PATH=MAE/CelebAMaskHQ_2D ## AbdAtlas1000, Abdomen1k
CLASSES=5 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16 ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v0
TRAIN_Ver=v0
LOSS_Ver=L2 ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
##################################################################

## for face
#########################
#### Compare Step1: Pretrain/mae ####
#########################
# CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio ${MASK_RATIO} --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt

########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=3
ckpt=1199
THRE=(0.02) # only face use 0.02
CLS=(',')
MASK=(0.0 0.25 0.75) ## 0.2 0.4 0.6 0.8
for cls_i in {0..0..1} ## {0..12..1}
do
for thre in {0..0..1}
do
for mask in {0..2..1}
do
CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_2D_gen.py --model ${MODEL_NAME} --mask_ratio ${MASK[${mask}]} --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}_maskrm${MASK[${mask}]}.txt
done
done
done






