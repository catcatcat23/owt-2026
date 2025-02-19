# CUDA_VISIBLE_DEVICES=0 bash eval_run1.sh

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

########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=0
ckpt=1199
THRE=(0.02 0.15)
CLS=(',' '0' '0,1' '0,2' '0,3' '0,4' '0,1,3' '0,1,4' '0,2,3' '0,2,4' '0,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '0,1,2' '1' '2' '3' '4' '1,2,3,4')
# CLS=(',' '0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
# CLS=(',' '0' '1' '2' '3' '4' '5' '1,2,3,4,5' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4')
for cls_i in {0..20..1} ## {0..12..1}
do
for thre in {0..1..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done


########################################################
#### Step3: seg evaluation ####
########################################################
#### Abdomen 1k ####
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=0
THRE=(0.02 0.15) # (0.25) (0.25 0.3 0.35 0.4)
CLS=('0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for cls_i in {0..9..1} # {0..4..1} # {0..5..1}
do
for thre in {0..1..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --thre ${THRE[${thre}]} --select_cls ${CLS[${cls_i}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done


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
TRAIN_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/image/
TRAIN_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training/mask/
TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/image/
TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test/mask/


########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=0
ckpt=1199
THRE=(0.02 0.15)
CLS=(',' '0' '0,1' '0,2' '0,3' '0,4' '0,1,3' '0,1,4' '0,2,3' '0,2,4' '0,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '0,1,2' '1' '2' '3' '4' '1,2,3,4')
# CLS=(',' '0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
# CLS=(',' '0' '1' '2' '3' '4' '5' '1,2,3,4,5' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4')
for cls_i in {0..20..1} ## {0..12..1}
do
for thre in {0..1..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done


########################################################
#### Step3: seg evaluation ####
########################################################
#### Abdomen 1k ####
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=0
THRE=(0.02 0.15) # (0.25) (0.25 0.3 0.35 0.4)
CLS=('0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for cls_i in {0..9..1} # {0..4..1} # {0..5..1}
do
for thre in {0..1..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --thre ${THRE[${thre}]} --select_cls ${CLS[${cls_i}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done
