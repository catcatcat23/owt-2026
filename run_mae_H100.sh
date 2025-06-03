## 同时，只有main_pretrain, OWC2, util/misc 有适配H100的debug，注意别和A100版本传错了

#########################
#### Step1: Pretrain ####
#########################
# CUDA_VISIBLE_DEVICES=${GPU_num} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port=${PORT_NUM} --use_env main_pretrain.py --batch_size ${BATCH_SIZE} --model ${MODEL_NAME} --mask_ratio 1.0 --epochs ${EPOCH} --warmup_epochs ${WARMUP} --blr ${BASE_LR} --weight_decay 0.05 --input_size ${INPUT_SIZE} --num_classes ${CLASSES} --arch_version ${ARCH_Ver} --training_version ${TRAIN_Ver} --token_factor ${TOKEN_Fac} --loss_version ${LOSS_Ver} --data_path ${TRAIN_CSV_PATH} --output_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --log_dir ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/ --save_freq 100 > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/Result.txt

########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=6
ckpt=1199
THRE=(0.02 0.15) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
# CLS=(',' '0' '0,1' '0,2' '0,3' '0,4' '0,1,3' '0,1,4' '0,2,3' '0,2,4' '0,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '0,1,2' '1' '2' '3' '4' '1,2,3,4')
# CLS=(',' '0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
# CLS=(',' '0' '1' '2' '3' '4' '5' '1,2,3,4,5' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4')
CLS=(',' '0' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for cls_i in {1..6..1} ## {0..20..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done

########################################################
#### Step3: seg evaluation ####
########################################################
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=0
THRE=(0.02 0.15) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=('0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
for cls_i in {0..9..1} # {0..4..1} # {0..5..1}
do
for thre in {1..1..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 0 --thre ${THRE[${thre}]} --select_cls ${CLS[${cls_i}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done

#########################################################################
#### Step2-other: Save inter feature (only use for latent diffusion) ####
#########################################################################
GPU=0
for ckpt in {1199..1199..100} # {100..1100..100}
do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TRAIN_DATA_PATH} --load_label_vis_path ${TRAIN_LABL_PATH} --load_csv_type train --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_train_epoch${ckpt}_cls''.txt
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_test_epoch${ckpt}_cls''.txt
done

##############################################################
#### Step4-other: load inter feature, for basic retrieval ####
##############################################################
ckpt=1199
GPU=0
# id_index_train=170
# id_index_test=0
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/
# rm ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/${id_index_train}.txt
# rm ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/${id_index_test}.txt
# echo " " > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/${id_index_train}.txt
# echo " " > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/${id_index_test}.txt
CLS=(',' '0' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '1,2,3,4')
for id_index_train in {6..6..2}
do
for topk in {2..2..2} ## top3, top5, top7
do
for cls_i in {0..0..1}
do
CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_retrieval.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TRAIN_DATA_PATH} --load_label_vis_path ${TRAIN_LABL_PATH} --load_csv_type train --save_video 0 --tnse_plot 0 --select_cls ${CLS[${cls_i}]} --topk ${topk} --id_index ${id_index_train} #>> ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/${id_index_train}.txt
# CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_retrieval.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --tnse_plot 0 --select_cls ${CLS[${cls_i}]} --topk ${topk} --id_index ${id_index_test} >> ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/${id_index_test}.txt
done
done
done



########################################################
Pure2D######################################################## ## (current only face)
########################################################
#### Step2: gen evaluation, better performance using vis_OWC2_LIB_3D_gen) ####
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=0
ckpt=1199
THRE=(0.02) # only face use 0.02
CLS=(',' '0' '0,1' '0,2' '0,3' '0,4' '0,5' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4' '1,2,3,4,5')
for cls_i in {0..12..1} ## {0..12..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_2D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done
########################################################
#### Step3: seg evaluation #### currently only face
########################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
ckpt=1199
GPU=0
THRE=(0.02) # only face use 0.02
CLS=('0' '1' '2' '3' '4' '5' '1,2,3,4,5' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4')
for cls_i in {0..11..1} # {0..4..1} # {0..5..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_2D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --thre ${THRE[${thre}]} --select_cls ${CLS[${cls_i}]} > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done

#########################################################################
#### Step2-other: Save inter feature (only use for latent diffusion) ####
#########################################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
GPU=7
ckpt=1199
# CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_2D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TRAIN_DATA_PATH} --load_label_vis_path ${TRAIN_LABL_PATH} --load_csv_type train --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_train_epoch${ckpt}_cls''.txt
CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_2D_gen_fix.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls '' > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen_test_epoch${ckpt}_cls''.txt

##############################################################
#### Step4-other: load inter feature, for basic retrieval ####
##############################################################
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
ckpt=1199
GPU=0
# id_index_train=0
# id_index_test=10
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/
# rm ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/${id_index_train}.txt
# rm ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/${id_index_test}.txt
# echo " " > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/${id_index_train}.txt
# echo " " > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/${id_index_test}.txt
CLS=(',' '0' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4')
# for id_index_train in {0..0..10}
for id_index_test in {0..80..4}
do
for topk in {1..1..1} ## top3, top5, top7
do
for cls_i in {0..6..1}
do
# CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_2D_retrieval.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TRAIN_DATA_PATH} --load_label_vis_path ${TRAIN_LABL_PATH} --load_csv_type train --save_video 1 --select_cls ${CLS[${cls_i}]} --topk ${topk} --id_index ${id_index_train} >> ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/train/${id_index_train}.txt
CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_2D_retrieval.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 1 --select_cls ${CLS[${cls_i}]} --topk ${topk} --id_index ${id_index_test} #>> ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/generate_in_step4/test/${id_index_test}.txt
done
done
done




CLS=(',' '0' '0,1' '0,2' '0,3' '0,4' '0,1,2' '0,1,3' '0,1,4' '0,2,3' '0,2,4' '0,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '1' '2' '3' '4' '1,2,3,4')
################################
#### integrate csv of gen ####
################################
cd /raid/home/CAMCA/ss3112/Github/mae/datasets
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/
## whole image
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls,_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/whole_image.txt
## 1 organ
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls0,2,3,4_0.02.txt; eval_gen_test_epoch1199_cls0,1,3,4_0.02.txt; eval_gen_test_epoch1199_cls0,1,2,4_0.02.txt; eval_gen_test_epoch1199_cls0,1,2,3_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/1_organ.txt 
## 2 organs
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls0,1,2_0.02.txt; eval_gen_test_epoch1199_cls0,1,3_0.02.txt; eval_gen_test_epoch1199_cls0,1,4_0.02.txt; eval_gen_test_epoch1199_cls0,2,3_0.02.txt; eval_gen_test_epoch1199_cls0,2,4_0.02.txt; eval_gen_test_epoch1199_cls0,3,4_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/2_organ.txt 
## 3 organs
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls0,1_0.02.txt; eval_gen_test_epoch1199_cls0,2_0.02.txt; eval_gen_test_epoch1199_cls0,3_0.02.txt; eval_gen_test_epoch1199_cls0,4_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/3_organ.txt
## 4 organs
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls0_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/4_organ.txt
## bg
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls1,2,3,4_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/bg.txt


CLS=('0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
################################
#### integrate csv of seg ####
################################
cd /raid/home/CAMCA/ss3112/Github/mae/datasets
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_seg/
## 1 organ
python integrate_csv_for_seg.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/ "eval_seg_test_epoch1199_cls0,2,3,4_0.15.txt; eval_seg_test_epoch1199_cls0,1,3,4_0.15.txt; eval_seg_test_epoch1199_cls0,1,2,4_0.15.txt; eval_seg_test_epoch1199_cls0,1,2,3_0.15.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_seg/1_organ_seg.txt 
## 4 organ (indirect 1 organ seg)
python integrate_csv_for_seg.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/ "eval_seg_test_epoch1199_cls1_0.15.txt; eval_seg_test_epoch1199_cls2_0.15.txt; eval_seg_test_epoch1199_cls3_0.15.txt; eval_seg_test_epoch1199_cls4_0.15.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_seg/1_organ_seg_indirect.txt 

################################################
#### corss dataset integrate csv of gen/seg ####
################################################
cd /raid/home/CAMCA/ss3112/Github/mae/datasets
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/
## only cross
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "cross_eval_gen_test_epoch1199_cls0,2,3,4_0.02.txt; cross_eval_gen_test_epoch1199_cls0,1,3,4_0.02.txt; cross_eval_gen_test_epoch1199_cls0,1,2,4_0.02.txt; cross_eval_gen_test_epoch1199_cls0,1,2,3_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/cross_1_organ.txt 
python integrate_csv_for_seg.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/ "cross_eval_seg_test_epoch1199_cls0,2,3,4_0.15.txt; cross_eval_seg_test_epoch1199_cls0,1,3,4_0.15.txt; cross_eval_seg_test_epoch1199_cls0,1,2,4_0.15.txt; cross_eval_seg_test_epoch1199_cls0,1,2,3_0.15.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_seg/cross_1_organ_seg.txt 


####################################
#### FACE integrate csv of gen ####
####################################
cd /raid/home/CAMCA/ss3112/Github/mae/datasets
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/
## whole image
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls,_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/whole_image.txt
## 1 TG
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls0,2,3,4,5_0.02.txt; eval_gen_test_epoch1199_cls0,1,3,4,5_0.02.txt; eval_gen_test_epoch1199_cls0,1,2,4,5_0.02.txt; eval_gen_test_epoch1199_cls0,1,2,3,5_0.02.txt; eval_gen_test_epoch1199_cls0,1,2,3,4_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/1_organ.txt 
## G TG
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls0_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/4_organ.txt

####################################
#### FACE integrate csv of seg ####
####################################
cd /raid/home/CAMCA/ss3112/Github/mae/datasets
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_seg/
## 1 TG
python integrate_csv_for_seg.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/ "eval_seg_test_epoch1199_cls0,2,3,4,5_0.02.txt; eval_seg_test_epoch1199_cls0,1,3,4,5_0.02.txt; eval_seg_test_epoch1199_cls0,1,2,4,5_0.02.txt; eval_seg_test_epoch1199_cls0,1,2,3,5_0.02.txt; eval_seg_test_epoch1199_cls0,1,2,3,4_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_seg/1_organ_seg.txt 
## 5 TG (indirect 1 TG seg)
python integrate_csv_for_seg.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/ "eval_seg_test_epoch1199_cls1_0.02.txt; eval_seg_test_epoch1199_cls2_0.02.txt; eval_seg_test_epoch1199_cls3_0.02.txt; eval_seg_test_epoch1199_cls4_0.02.txt; eval_seg_test_epoch1199_cls5_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_seg/1_organ_seg_indirect.txt 






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


# ########################################################
# #### Step3: seg evaluation ####
# ########################################################
# PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
# TEST_DATA_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Others/P1_C1/Test/image/
# TEST_LABL_PATH=/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Others/P1_C1/Test/mask/
# mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/
# ckpt=1199
# GPU=4
# THRE=(0.25) # (0.25) (0.25 0.3 0.35 0.4)
# for cls in {0..9..1}
# do
# for thre in {0..0..1}
# 	do
# 	CUDA_VISIBLE_DEVICES=${GPU} python vis_OWC2_LIB_3D_seg.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --thre ${THRE[${thre}]} --select_cls ${cls} #> ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_seg/eval_seg_test_epoch${ckpt}_cls${cls}_${THRE[${thre}]}.txt
# done
# done

## very tmp 3d gen with print slices (--save_video 2 and --id_index 5)
PROCESS_NAME=Token_${MODEL_NAME}_${BASE_LR}_${CLASSES}_${INPUT_SIZE}_${TOKEN_Fac}_${ARCH_Ver}_${TRAIN_Ver}_${LOSS_Ver}_GPU${N_GPU}_${BATCH_SIZE}_${EPOCH}
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/
GPU=0
ckpt=1199
THRE=(0.02 0.15) ## 只有含0的(direct)至少去掉底噪，或者0.15 (根据数据集决定)
CLS=(',' '0' '0,1' '0,2' '0,3' '0,4' '0,1,3' '0,1,4' '0,2,3' '0,2,4' '0,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3' '0,1,2' '1' '2' '3' '4' '1,2,3,4')
# CLS=(',' '0' '1' '2' '3' '4' '1,2,3,4' '0,2,3,4' '0,1,3,4' '0,1,2,4' '0,1,2,3')
# CLS=(',' '0' '1' '2' '3' '4' '5' '1,2,3,4,5' '0,2,3,4,5' '0,1,3,4,5' '0,1,2,4,5' '0,1,2,3,5' '0,1,2,3,4')
for cls_i in {0..20..1} ## {0..12..1}
do
for thre in {0..0..1}
	do
	CUDA_VISIBLE_DEVICES=${GPU} OMP_NUM_THREADS=1 python vis_OWC2_LIB_3D_gen.py --model ${MODEL_NAME} --mask_ratio 1.0 --input_size 224 --arch_version ${ARCH_Ver} --num_classes ${CLASSES} --token_factor ${TOKEN_Fac} --training_version ${TRAIN_Ver} --loss_version ${LOSS_Ver} --output_vis ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/vis/ --checkpoint ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/checkpoint-${ckpt}.pth --reverse 0 --load_data_vis_path ${TEST_DATA_PATH} --load_label_vis_path ${TEST_LABL_PATH} --load_csv_type test --save_video 2 --select_cls ${CLS[${cls_i}]} --thre ${THRE[${thre}]} --id_index 5 #> ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/eval_gen_test_epoch${ckpt}_cls${CLS[${cls_i}]}_${THRE[${thre}]}.txt
done
done


cd /raid/home/CAMCA/ss3112/Github/mae/datasets
mkdir -p ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/
## 1 organ
python integrate_csv_for_gen.py ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/eval_gen/ "eval_gen_test_epoch1199_cls1_0.02.txt; eval_gen_test_epoch1199_cls2_0.02.txt; eval_gen_test_epoch1199_cls3_0.02.txt; eval_gen_test_epoch1199_cls4_0.02.txt" > ${RESULT_PATH}/${CUR_PATH}/${PROCESS_NAME}/inte_eval_gen/1_organ.txt 
