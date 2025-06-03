# ############################################################################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25674
# GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=6
# BASE_LR=1e-4 #4e-4 #1.5e-5 #8e-5 ## 1e-4 1.5e-4 5e-5 4.5e-6
# INPUT_SIZE=224
# TOKEN_Fac=20 #200 #20 for 2d 200 for 3d
# ####################################################
# CUR_PATH=AbdAtlas1000_3D ## AbdAtlas1000, Abdomen1k
# CLASSES=9 ## 9 for AbdAtlas1000; 5 for Abdomen1k
# ####################################################
# MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=64 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB); 1 for 3D (temp); 48 for -LA -Fixfr16; 12 for -Fixfr16 (w/o -LA)
# EPOCH=1000 ## v1: 800-40; v2: 400-20
# WARMUP=50 ## v1: 800-40; v2: 400-20
# ARCH_Ver=v11
# ####     v1 (Trans Decoder), v2 (VQGAN Decoder), v3 (VQGAN Encoder to 1/16hw before two transformers);
# ####         v11; v31: without intermediate cls tokens
# ####         -cls4 (用于skip的 slice/global information) (currently, only for v3)
# ####     -VQv0/v01/v1_nt512; (v01比v0用的non-linear; v1用的NormEMA QT)
# ####         -LIBv0 (是否一个organ一个codebook); -DTv0;
# TRAIN_Ver=v01-3D-Fixfr4-TS1 ## v0 (generation) (v01 generation without mask tokens); v1 (inpainting); v01-3D-Fixfr4-TS1
# ####     -3D -Fixfr16
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv
# # TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv !!!! 10000*pay attention !!!! 5 labels may cause wrong mapping
# # TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr16_H100_2_Final112.csv ## !!!! 10000*pay attention !!!! 5 labels may cause wrong mapping
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# # PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}

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
# ###################################################################
# ## AbdAutoPet final 16 slices -v31 (not yet 4.5 days)
# ###################################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25666
# GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=6
# BASE_LR=1e-4
# INPUT_SIZE=224
# TOKEN_Fac=20
# ####################################################
# CUR_PATH=AbdAutoPet_3D ## AbdAtlas1000, Abdomen1k
# CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
# ####################################################
# MODEL_NAME=mae_vit_basefix16_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=8
# EPOCH=600
# WARMUP=30
# ARCH_Ver=v31
# TRAIN_Ver=v01-3D-Fixfr16-TS1
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr16_H100_2_Final112.csv
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
# ####################################################################
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
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
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
##############################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/mask/
##############################################################
## Delay final 2D (MRI 2D) (finished)
##############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=SyntheticMRI_Delay_2D ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_H100_2_Final112.csv
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
############################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/mask/
############################################################
## PreArtery final 2D (MRI 2D) (finished)
############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25668
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=SyntheticMRI_PreArtery_2D ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################


## Real Natural Image 2D
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





## Inpainting Experiments
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_Inpainting ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v11
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
###########################################################
## Abdomen1k_3D final 1 slice (2D-Trans) (finished)
###########################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25638
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/Abdomen1k_2D_Inpainting ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96 
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v11
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv
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
##############################################################
## Delay final 2D (MRI 2D) (finished)
##############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/SyntheticMRI_Delay_2D_Inpainting ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v11
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/mask/
############################################################
## PreArtery final 2D (MRI 2D) (finished)
############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25668
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/SyntheticMRI_PreArtery_2D_Inpainting ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v11
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/mask/


## only for inpainting eval ## 
########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=4
ckpt=1199
THRE=(0.02)
CLS=('1' '2' '3' '4') ## only for inpainting 只能预测器官（不能包含背景）
for cls_i in {0..3..1} ## {0..12..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done



## Ablation Studies: Data percentage
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25636
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_20data ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112_rand140.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25646
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_60data ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112_rand420.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################


## ICCV Rebuttal ##
## Ablation Studies: Data percentage (reload mae/LPIPS)
## 首先预训练设置 mask ratio = 0.0, 做无监督训练
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25666
GPU_num='4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=3
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_00_Unsupervised_v12 ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=96
EPOCH=600 ##1200
WARMUP=30 ##60
ARCH_Ver=v12unsup ## v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 0.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 400 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt
## 再 load unsupervised 去做 hybrid supervised
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25636
GPU_num='0,1' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_20data_RELOAD_v12 ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112_rand140.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25656
GPU_num='2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=5
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_60data_RELOAD_v12 ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112_rand420.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25646
GPU_num='0,1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=7
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_100data_RELOAD_v12 ## AbdAtlas1000, Abdomen1k
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

## only for RELOAD
# reload_ckpt=/raid/home/CAMCA/ss3112/Github/mae/Results/Ablation/AbdAutoPet_2D_00_Unsupervised/Token_mae_vit_base_patch16-LA_1e-4_4_224_20_v11_v01_L2-LPIPS_GPU4_96_1200/checkpoint-1199.pth
reload_ckpt=/raid/home/CAMCA/ss3112/Github/mae/Results/Ablation/AbdAutoPet_2D_00_Unsupervised_v12/Token_mae_vit_base_patch16-LA_1e-4_4_224_20_v12unsup_v01_L2-LPIPS_GPU3_96_600/checkpoint-599.pth
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 200 --checkpoint ${reload_ckpt} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt


## Ablation Studies: Token Number
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25616
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=5
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
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
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25616
GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=10
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
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
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25626
GPU_num='6,7' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=40
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D ## AbdAtlas1000, Abdomen1k
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



##################################################################
only-Text-encoding (CLIP)
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25668
GPU_num='0,3,4' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=3
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_text ## AbdAtlas1000, Abdomen1k
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
TEXT_ENC="./txt_encoding.pth"
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
##################################################################
## ICCV Rebuttal ##
only-Text-encoding (CT_CLIP)
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25668
GPU_num='4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=3
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_text_CT_CLIP ## AbdAtlas1000, Abdomen1k
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
TEXT_ENC="./txt_encoding_CT_CLIP.pth"
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/
##################################################################
## AbdAutoPet final 1 slice (2D-Trans) (finished)
##################################################################
cd /raid/camca/ss3112/Github/mae/
PORT_NUM=25668
GPU_num='0,1,2,3,4,5' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_text_CT_CLIP_full ## AbdAtlas1000, Abdomen1k
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
TEXT_ENC="./txt_encoding_CT_CLIP_full.pth"
####################################################
TRAIN_CSV_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_1_Final112.csv
RESULT_PATH=/raid/camca/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
################################################################
TRAIN_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/

# ###########################################################
# ## Abdomen1k_3D final 1 slice (2D-Trans) (finished)
# ###########################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25694
# GPU_num='5,6,7' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=3
# BASE_LR=1e-4
# INPUT_SIZE=224
# TOKEN_Fac=20
# ####################################################
# CUR_PATH=Ablation/Abdomen1k_2D_text ## AbdAtlas1000, Abdomen1k
# CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
# ####################################################
# MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=96 
# EPOCH=1200
# WARMUP=60
# ARCH_Ver=v11
# TRAIN_Ver=v01
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# TEXT_ENC="./txt_encoding.pth"
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
# ###########################################################
# TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
# TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
# TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
# TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/


##############################################################
## Delay final 2D (MRI 2D) (finished)
##############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25633
GPU_num='0,3' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/SyntheticMRI_Delay_2D_text ## AbdAtlas1000, Abdomen1k
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
TEXT_ENC="./txt_encoding_delay.pth"
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/mask/
##############################################################
## ICCV Rebuttal ##
only-Text-encoding (CT_CLIP)
##############################################################
## Delay final 2D (MRI 2D) (finished)
##############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25633
GPU_num='2,3' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/SyntheticMRI_Delay_2D_text_CT_CLIP ## AbdAtlas1000, Abdomen1k
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
TEXT_ENC="./txt_encoding_CT_CLIP_Delay.pth"
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/mask/


# ##############################################################
# ## PreArtery final 2D (MRI 2D) (finished)
# ##############################################################
# cd /raid/home/CAMCA/ss3112/Github/mae/
# PORT_NUM=25668
# GPU_num='4,5' # '0,1,2,3,4,5' '4,5,6,7'
# N_GPU=2
# BASE_LR=1e-4
# INPUT_SIZE=224
# TOKEN_Fac=20
# ####################################################
# CUR_PATH=Ablation/SyntheticMRI_PreArtery_2D_text ## AbdAtlas1000, Abdomen1k
# CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
# ####################################################
# MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
# ####	 -LA (linear attention)
# BATCH_SIZE=96
# EPOCH=1200
# WARMUP=60
# ARCH_Ver=v11
# TRAIN_Ver=v01
# LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
# TEXT_ENC="./txt_encoding_delay.pth"
# ####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
# RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
# ############################################################################################################
# TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/image/
# TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/mask/
# TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/image/
# TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/mask/



CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 --text_encoding ${TEXT_ENC} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt

## only text step2
########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=5
ckpt=1199
THRE=(0.02 0.15) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=(',' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
# CLS=(',' '0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
# CLS=(',' '0' '1' '2' '3' '4' '5' '1,2,3,4,5' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4')
for cls_i in {0..4..1} ## {0..12..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} --text_encoding ${TEXT_ENC} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done

## only text step3
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=3
THRE=(0.02 0.15) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=('0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for cls_i in {0..3..1} # {0..4..1} # {0..5..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --thre ${THRE[${thre}]} --select_cls ${CLS[${cls_i}]} --text_encoding ${TEXT_ENC} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done
























































############################################################
## PreArtery final 2D (MRI 2D) (finished)
############################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25668
GPU_num='5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=2
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=bak ## AbdAtlas1000, Abdomen1k
CLASSES=4 ## 9 for AbdAtlas1000; 4 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_base_patch16-LA ## mae_vit_base_patch16, mae_vit_basefix16_patch16-LA
####	 -LA (linear attention)
BATCH_SIZE=16
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################


