## 同时，只有main_pretrain, OWC2, util/misc 有适配H100的debug，注意别和A100版本传错了
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
PORT_NUM=25664
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
############################################################################################################
############################################################################################################
## Abdomen1k_3D final 4 slices (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25664
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
############################################################################################################
############################################################################################################
## Abdomen1k_3D final 1 slice (2D) (running)
############################################################################################################
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
############################################################################################################
############################################################################################################

############################################################################################################
## Abdomen1k_3D final 4 slices (Tumor) (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25664
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Abdomen1k_Tumor_3D ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_Fixfr4_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
############################################################################################################
## Abdomen1k_3D final 1 slice (2D) (Tumor) (running)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25694
GPU_num='1,2,3,4,5,6' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=6
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Abdomen1k_Tumor_2D ## AbdAtlas1000, Abdomen1k
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
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_H100_2_Final112.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
############################################################################################################

############################################################################################################
## AbdAtlas final 4 slices (4c) (not yet)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25664
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
############################################################################################################
############################################################################################################
## AbdAtlas final 1 slice (2D) (4c) (not yet)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25664
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
############################################################################################################

############################################################################################################
## AbdAutoPet final 4 slices (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25664
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
############################################################################################################
############################################################################################################
## AbdAutoPet final 1 slice (2D) (finished)
############################################################################################################
cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25664
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
############################################################################################################

#########################
#### Step1: Pretrain ####
#########################
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt

########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=3
# for ckpt in {1199..1199..100} # {100..1100..100}
# do
ckpt=1199
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls''.txt
# done

ckpt=1199
CLS=('0' '1' '2' '3' '4' '1,2' '1,3' '1,4' '2,3' '2,4' '3,4' '1,2,3' '1,2,4' '1,3,4' '2,3,4' '1,2,3,4')
CLS=('0' '1' '2' '3' '4' '5' '1,2' '1,3' '1,4' '2,3' '2,4' '3,4' '1,2,3' '1,2,4' '1,3,4' '2,3,4' '1,2,3,4')
for cls_i in {0..15..1}
do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}.txt
done

########################################################
#### Step3: seg evaluation ####
########################################################
#### Abdomen 1k ####
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
# TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
# TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Test/mask/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=0
THRE=(0.25) # (0.25) (0.25 0.3 0.35 0.4)
for cls in {0..5..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --thre ${THRE[${thre}]} --select_cls ${cls} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${cls}_${THRE[${thre}]}.txt
done
done

#### Atlas ####
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test/mask/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=2
THRE=(0.25) # (0.25) (0.25 0.3 0.35 0.4)
for cls in {2..9..7}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --thre ${THRE[${thre}]} --select_cls ${cls} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${cls}_${THRE[${thre}]}.txt
done
done

#########################################################################
#### Step2-other: Save inter feature (only use for latent diffusion) ####
#########################################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/
# ckpt=800
GPU=5
for ckpt in {1199..1199..100} # {100..1100..100}
do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TRAIN_DATA_PATH} --load_label_vis_path ${TRAIN_LABL_PATH} --load_csv_type train --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_train_epoch${ckpt}_cls''.txt
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_test_epoch${ckpt}_cls''.txt
done




## vis
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
ckpt=600
GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '' --reverse 0 --sim_print True
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '1' --reverse 1
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '2' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '2' --reverse 1
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '0,1,3,4,5,6,7,8,9' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '0,1,2,3,4,6,7,8,9' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '0,1,2,3,4,6,7,8,9' --reverse 1
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '3' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '3' --reverse 1
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '5' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '5' --reverse 1
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '5,8,9' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '5,8,9' --reverse 1
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '1,2,3,4,5,6,7,8,9' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '1,2,3,4,5,6,7,8,9' --reverse 1
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '0' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '0' --reverse 1

PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
ckpt=100
PORT_NUM_test=25655
GPU=5
BATCH_SIZE_test=1
N_GPU_test=1
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU_test} --master_port=${PORT_NUM_test} --use_env eval_pretrain.py --batch_size ${BATCH_SIZE_test} --model ${MODEL_NAME} --mask_ratio 0.5 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 --resume ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/


########################################################
#### Step3: seg evaluation ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Others/P1_C1/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Others/P1_C1/Test/mask/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=4
THRE=(0.25) # (0.25) (0.25 0.3 0.35 0.4)
for cls in {0..9..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --thre ${THRE[${thre}]} --select_cls ${cls} #> ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${cls}_${THRE[${thre}]}.txt
done
done
