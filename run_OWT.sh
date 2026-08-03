#######################
#### Main Training ####
#######################
## 3D datasets (Abdomen1k, AutoPet, RAOS/Delay, RAOS/PreArtery)
## 4-slice training for medical image datasets ## 
####################################################
SERVER_PATH=/raid/camca/ss3112 ## H100_1
cd ${SERVER_PATH}/Github/OWT/
####################################################
PORT_NUM=25666
GPU_num='4,5,6'
N_GPU=3
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
TRAIN_CSV_PATH=${SERVER_PATH}/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr4_H100_1_Final112.csv
RESULT_PATH=${SERVER_PATH}/Github/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/

## 1-slice training for medical image datasets ## 
####################################################
SERVER_PATH=/raid/camca/ss3112 ## H100_1
cd ${SERVER_PATH}/Github/OWT/
####################################################
PORT_NUM=25666
GPU_num='2,3,4,5,6,7'
N_GPU=6
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
RESULT_PATH=${SERVER_PATH}/Github/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Training/image
TRAIN_LABL_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Training/mask
TEST_DATA_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Test/image/
TEST_LABL_PATH=/gpfs/work/aac/bolinren19/2026-07/DATA/zip/Test/mask/

## 1-slice alter with 2D VAGAN Encoder/Decoder, training for medical image datasets ## 
####################################################
SERVER_PATH=/raid/camca/ss3112 ## H100_1
cd ${SERVER_PATH}/Github/OWT/
####################################################
PORT_NUM=25666
GPU_num='4,5,6,7'
N_GPU=4
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=AbdAutoPet_2D 
CLASSES=4 
####################################################
MODEL_NAME=mae_vit_base_patch16-LA  ## -LA (linear attention)
BATCH_SIZE=32 
EPOCH=1200
WARMUP=60
ARCH_Ver=v31
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; -LPIPS
####################################################
TRAIN_CSV_PATH=${SERVER_PATH}/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_1_Final112.csv
RESULT_PATH=${SERVER_PATH}/Github/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/

#########################
#### Step1: Training ####
#########################
# CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt


#######################################
#### Step2: gen evaluation (2D/3D) ####
#######################################
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=6
ckpt=1199
CLS=(',' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for cls_i in {4..4..1}
do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls ${CLS[${cls_i}]} --thre 0.02 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_0.02.txt
done

#######################################
#### Step3: seg evaluation (2D/3D) ####
#######################################
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=7
CLS=('0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '1' '2' '3' '4')
for cls_i in {0..7..1}
do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --thre 0.15 --select_cls ${CLS[${cls_i}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${CLS[${cls_i}]}_0.15.txt
done

################################################################################################
#### Step4-1: Save inter feature (use only for experiments training with more than 1-slice) ####
################################################################################################
GPU=7
ckpt=1199
## Training
CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TRAIN_DATA_PATH} --load_label_vis_path ${TRAIN_LABL_PATH} --load_csv_type train --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_train_epoch${ckpt}_cls''.txt
## Test
CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_test_epoch${ckpt}_cls''.txt

################################################################################################
#### Step4-2: t-sne and retrieval (use only for experiments training with more than 1-slice) ####
################################################################################################
ckpt=1199
GPU=7
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/
CLS=(',' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for id_index_ds in {6..6..1}
do
for topk in {2..2..2} ## top2, top3, top5, top7
do
for cls_i in {0..0..1}
do
# CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_retrieval.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TRAIN_DATA_PATH} --load_label_vis_path ${TRAIN_LABL_PATH} --load_csv_type train --save_video 0 --tnse_plot 1 --select_cls ${CLS[${cls_i}]} --topk ${topk} --id_index ${id_index_ds} 
CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_retrieval.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --tnse_plot 1 --select_cls ${CLS[${cls_i}]} --topk ${topk} --id_index ${id_index_ds}
done
done
done


## 2D dataset (CelebAMaskHQ)
## 1-slice training for natural image dataset ## 
####################################################
SERVER_PATH=/raid/camca/ss3112 ## H100_1
cd ${SERVER_PATH}/Github/OWT/
####################################################
PORT_NUM=25666
GPU_num='4,5,6,7'
N_GPU=4
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=CelebAMaskHQ_2D 
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
TRAIN_CSV_PATH=${SERVER_PATH}/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/Training_H100_1_Final112.csv
RESULT_PATH=${SERVER_PATH}/Github/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/image/
TRAIN_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/mask/
TEST_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Test/image/
TEST_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Test/mask/

## 1-slice alter with 2D VAGAN Encoder/Decoder, training for natural image dataset ## 
####################################################
SERVER_PATH=/raid/camca/ss3112 ## H100_1
cd ${SERVER_PATH}/Github/OWT/
####################################################
PORT_NUM=25666
GPU_num='4,5,6,7'
N_GPU=4
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=CelebAMaskHQ_2D 
CLASSES=4 
####################################################
MODEL_NAME=mae_vit_base_patch16-LA  ## -LA (linear attention)
BATCH_SIZE=32 
EPOCH=1200
WARMUP=60
ARCH_Ver=v31
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; -LPIPS
####################################################
TRAIN_CSV_PATH=${SERVER_PATH}/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/Training_H100_1_Final112.csv
RESULT_PATH=${SERVER_PATH}/Github/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/image/
TRAIN_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Training/mask/
TEST_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Test/image/
TEST_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Other2D/CelebAMask-HQ/s02_Final/Test/mask/

## Training
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt


















########################
#### Other Training ####
########################
## 1-slice training for medical image datasets ## 
####################################################
SERVER_PATH=/raid/camca/ss3112 ## H100_1
cd ${SERVER_PATH}/Github/OWT/
####################################################
PORT_NUM=25666
GPU_num='4,5,6,7'
N_GPU=4
BASE_LR=1e-4
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
CUR_PATH=Ablation/AbdAutoPet_2D_text_CT_CLIP 
CLASSES=4 
####################################################
MODEL_NAME=mae_vit_base_patch16-LA  ## -LA (linear attention)
BATCH_SIZE=64 
EPOCH=1200
WARMUP=60
ARCH_Ver=v11
TRAIN_Ver=v01
LOSS_Ver=L2-LPIPS ## L2; -LPIPS
TEXT_ENC="./datasets/txt_encoding_CT_CLIP.pth"
####################################################
TRAIN_CSV_PATH=${SERVER_PATH}/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_1_Final112.csv
RESULT_PATH=${SERVER_PATH}/Github/OWT/Results/
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
###########################################################
TRAIN_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/image/
TRAIN_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/mask/
TEST_DATA_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/image/
TEST_LABL_PATH=/raid/camca/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/mask/

CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 --text_encoding ${TEXT_ENC} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt

####################################################
#### Step2: gen evaluation (with text encoding) ####
####################################################
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=6
ckpt=1199
CLS=(',' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for cls_i in {4..4..1}
do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} --thre 0.02 --text_encoding ${TEXT_ENC} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_0.02.txt
done

####################################################
#### Step3: seg evaluation (with text encoding) ####
####################################################
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=7
CLS=('0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '1' '2' '3' '4')
for cls_i in {0..7..1}
do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --thre 0.15 --select_cls ${CLS[${cls_i}]} --text_encoding ${TEXT_ENC} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${CLS[${cls_i}]}_0.15.txt
done





















