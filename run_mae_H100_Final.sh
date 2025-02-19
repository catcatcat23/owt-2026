############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25674
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4 #4e-4 #1.5e-5 #8e-5 ## 1e-4 1.5e-4 5e-5 4.5e-6
INPUT_SIZE=224
TOKEN_Fac=20 #200 #20 for 2d 200 for 3d
####################################################
CUR_PATH=AbdAtlas1000_3D ## AbdAtlas1000, Abdomen1k
CLASSES=9 ## 9 for AbdAtlas1000; 5 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=64 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB); 1 for 3D (temp); 48 for -LA -Fixfr16; 12 for -Fixfr16 (w/o -LA)
EPOCH=1000 ## v1: 800-40; v2: 400-20
WARMUP=50 ## v1: 800-40; v2: 400-20
ARCH_Ver=v11
####     v1 (Trans Decoder), v2 (VQGAN Decoder), v3 (VQGAN Encoder to 1/16hw before two transformers);
####         v11; v31: without intermediate cls tokens
####         -cls4 (用于skip的 slice/global information) (currently, only for v3)
####     -VQv0/v01/v1_nt512; (v01比v0用的non-linear; v1用的NormEMA QT)
####         -LIBv0 (是否一个organ一个codebook); -DTv0;
TRAIN_Ver=v01-3D-Fixfr4-TS1 ## v0 (generation) (v01 generation without mask tokens); v1 (inpainting); v01-3D-Fixfr4-TS1
####     -3D -Fixfr16
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv !!!! 10000*pay attention !!!! 5 labels may cause wrong mapping
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr16_H100_2_Final112.csv ## !!!! 10000*pay attention !!!! 5 labels may cause wrong mapping
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}

#########################
#### Command History ####
#########################
############################################################################################################
## Abdomen1k_3D final 16 slices (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4 #4e-4 #1.5e-5 #8e-5 ## 1e-4 1.5e-4 5e-5 4.5e-6
INPUT_SIZE=224
TOKEN_Fac=20 #200 #20 for 2d 200 for 3d
####################################################
CUR_PATH=Abdomen1k_3D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=20 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB); 1 for 3D (temp); 48 for -LA -Fixfr16; 12 for -Fixfr16 (w/o -LA)
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
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv !!!! 10000*pay attention !!!! 4 labels may cause wrong mapping
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr16_H100_2_Final112.csv ## !!!! 10000*pay attention !!!! 4 labels may cause wrong mapping
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
###########################################################
## Abdomen1k_3D final 4 slices (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Abdomen1k_3D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=64 
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr4-TS1 
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr4_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/
###########################################################
## Abdomen1k_3D final 1 slice (2D-Trans) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Abdomen1k_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
###########################################################

# ############################################################################################################
# ## Abdomen1k_3D final 4 slices (Tumor) (finished)
# ############################################################################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25666
# GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=6
# BASE_LR=1e-4
# INPUT_SIZE=224
# TOKEN_Fac=20
# ####################################################
# CUR_PATH=Abdomen1k_Tumor_3D ## AbdAtlas1000, Abdomen1k
# CLASSES=5 ## 9 for AbdAtlas1000; 4 for Abdomen1k
# ####################################################
# MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=64 
# EPOCH=1200
# WARMUP=60
# ARCH_Ver=v11
# TRAIN_Ver=v01-3D-Fixfr4-TS1 
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_Fixfr4_H100_2_Final112.csv
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
# ##############################################################
# ##############################################################
# ## Abdomen1k_3D final 1 slice (2D) (Tumor) (running)
# ##############################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25694
# GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=6
# BASE_LR=1e-4
# INPUT_SIZE=224
# TOKEN_Fac=20
# ####################################################
# CUR_PATH=Abdomen1k_Tumor_2D ## AbdAtlas1000, Abdomen1k
# CLASSES=5 ## 9 for AbdAtlas1000; 4 for Abdomen1k
# ####################################################
# MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=96 
# EPOCH=1200
# WARMUP=60
# ARCH_Ver=v11
# TRAIN_Ver=v01
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_H100_2_Final112.csv
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
# ###########################################################
# ###########################################################

############################################################################################################
## AbdAtlas final 4 slices (4c) (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAtlas1000_3D_4c ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=64
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr4-TS1
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_Fixfr4_H100_2_Final112_4c.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
#############################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training/mask4c/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test/mask4c/
#############################################################
## AbdAtlas final 1 slice (2D-Trans) (4c) (finished)
#############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAtlas1000_2D_4c ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_H100_2_Final112_4c.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###############################################################
###############################################################

############################################################################################################
## AbdAutoPet final 16 slices (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_3D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=20
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr16-TS1
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr16_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
###################################################################
## AbdAutoPet final 16 slices -v31 (not yet 4.5 days)
###################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_3D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=8
EPOCH=600
WARMUP=30
ARCH_Ver=v31
TRAIN_Ver=v01-3D-Fixfr16-TS1
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr16_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
####################################################################
####################################################################
## AbdAutoPet final 4 slices (finished)
####################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_3D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=64
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr4-TS1
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr4_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
##################################################################
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
################################################################
## AbdAutoPet final 1 slice (2D)-CNN ()
################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=32
EPOCH=1200
WARMUP=60
ARCH_Ver=v31
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################

############################################################################################################
## WAW final 4 slices (tumor 5c) (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=WAW_3D ## AbdAtlas1000, Abdomen1k
CLASSES=5 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=64
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr4-TS1
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/WAW-TACE/WAW_Final/Training/Training_Fixfr4_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/WAW-TACE/WAW_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/WAW-TACE/WAW_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/WAW-TACE/WAW_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/WAW-TACE/WAW_Final/Test/mask/
############################################################################################################


############################################################################################################
## Delay final 4 slices (MRI 4c) (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=4
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=SyntheticMRI_Delay_3D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=64
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr4-TS1
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_Fixfr4_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################

############################################################################################################
## PreArtery final 4 slices (MRI 4c) (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=5
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=SyntheticMRI_PreArtery_3D ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=64
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01-3D-Fixfr4-TS1
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_Fixfr4_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################


############################################################################################################
## Face final 1 slice (2D-Trans) (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25669
GPU_num='2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=5
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=CelebAMaskHQ_2D ## AbdAtlas1000, Abdomen1k
CLASSES=5 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01
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
##################################################################
## Face final 1 slice (2D-CNN) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25669
GPU_num='2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=5
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=CelebAMaskHQ_2D ## AbdAtlas1000, Abdomen1k
CLASSES=5 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=32
EPOCH=1200
WARMUP=60
ARCH_Ver=v31
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################


























