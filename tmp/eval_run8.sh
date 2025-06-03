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
##############################################################


########################################################
#### Compare Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=0
ckpt=1199
THRE=(0.02) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=(',')
for cls_i in {0..0..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 0.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done




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



########################################################
#### Compare Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}_${MASK_RATIO}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=0
ckpt=1199
THRE=(0.02) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=(',')
for cls_i in {0..0..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 0.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done
