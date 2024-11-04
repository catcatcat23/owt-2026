## 同时，只有main_pretrain, OWC2, util/misc 有适配H100的debug，注意别和A100版本传错了


cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25634
GPU_num='4,5,6,7' # '0,1,2,3' '4,5,6,7'
N_GPU=4
CUR_PATH=AbdAtlas1000
MODEL_NAME=mae_vit_base_patch16
BASE_LR=1e-4 #1.5e-5 #8e-5 ## 1e-4 1.5e-4 5e-5 4.5e-6
CLASSES=9
INPUT_SIZE=224
TOKEN_Fac=20
####################################################
BATCH_SIZE=32 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB)
EPOCH=800 ## v1: 800-40; v2: 400-20
WARMUP=40 ## v1: 800-40; v2: 400-20
ARCH_Ver=v31-cls4 ##v31-cls4-VQv01_nt512-LIBv01 
####     v1 (Trans Decoder), v2 (VQGAN Decoder), v3 (VQGAN Encoder to 1/16hw before two transformers);
####         -cls4 (用于skip的 slice/global information) (currently, only for v3)
####     -VQv0/v01/v1_nt512; (v01比v0用的non-linear; v1用的NormEMA QT)
####         -LIBv0 (是否一个organ一个codebook); -DTv0;
TRAIN_Ver=v0 ## v0 (generation); v1 (inpainting)
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training_H100_2.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
# CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt


## vis
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '3' --reverse 0
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '3' --reverse 1
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '5' --reverse 0
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '5' --reverse 1
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '6,8,9' --reverse 0
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '6,8,9' --reverse 1
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '1,2,3,4,5,6,7,8,9' --reverse 0
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '1,2,3,4,5,6,7,8,9' --reverse 1

CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '3,5,7,8,9' --reverse 0
CUDA_VISIBLE_DEVICES=0 python vis_OWC2_LIB.py --model mae_vit_base_patch16 --mask_ratio 0.5 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-500.pth --select_cls '3,5,7,8,9' --reverse 1




