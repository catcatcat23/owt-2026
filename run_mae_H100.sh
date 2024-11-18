## 同时，只有main_pretrain, OWC2, util/misc 有适配H100的debug，注意别和A100版本传错了


cd /raid/home/CAMCA/ss3112/Github/mae/
PORT_NUM=25644
GPU_num='0,1,2,3,4' # '0,1,2,3,4,5' '4,5,6,7'
N_GPU=5
BASE_LR=1e-4 #4e-4 #1.5e-5 #8e-5 ## 1e-4 1.5e-4 5e-5 4.5e-6
INPUT_SIZE=224
TOKEN_Fac=20 #200 #20 for 2d 200 for 3d
####################################################
CUR_PATH=AbdAtlas1000 ## AbdAtlas1000, Abdomen1k
CLASSES=9 ## 9 for AbdAtlas1000; 5 for Abdomen1k
####################################################
MODEL_NAME=mae_vit_basefix16_patch16 #-LA ## mae_vit_base_patch16_4_dim4 ()
####	 -LA (linear attention)
BATCH_SIZE=96 ## 64 for v1 in A100 40G (21401MiB); 16/24? for v2 in A100 40G (24693MiB); 1 for 3D (temp); 48 for -LA -Fixfr16; 12 for -Fixfr16 (w/o -LA)
EPOCH=1000 ## v1: 800-40; v2: 400-20
WARMUP=50 ## v1: 800-40; v2: 400-20
ARCH_Ver=v11
####     v1 (Trans Decoder), v2 (VQGAN Decoder), v3 (VQGAN Encoder to 1/16hw before two transformers);
####         v11; v31: without intermediate cls tokens
####         -cls4 (用于skip的 slice/global information) (currently, only for v3)
####     -VQv0/v01/v1_nt512; (v01比v0用的non-linear; v1用的NormEMA QT)
####         -LIBv0 (是否一个organ一个codebook); -DTv0;
TRAIN_Ver=v01-3D-Fixfr4 ## v0 (generation) (v01 generation without mask tokens); v1 (inpainting)
####     -3D -Fixfr16
LOSS_Ver=L2-LPIPS ## L2; L1; -LPIPS; -LPIPS-GAN
####################################################
# TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv
TRAIN_CSV_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Training_Fixfr16_H100_2.csv ##Training_H100_2.csv
RESULT_PATH=/raid/home/CAMCA/ss3112/Github/mae/Results/
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/
############################################################################################################
CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt


## vis
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
ckpt=300
GPU=5
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '2' --reverse 0
CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --num_classes 9 --arch_version ${ARCH_Ver} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --select_cls '0,1,3,4,5,6,7,8,9' --reverse 0
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





