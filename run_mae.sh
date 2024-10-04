
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/ImageNet/imagenet/

## BTCV
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 12 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med-SAM2D/ProMISe_Dataset/Datasets/synapseCT/Training/2D_all_5slice/training_full_path.csv #> tmp.txt & 

## AbdAtlas
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.5 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 9 --arch_version v1 --token_factor 20 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv > tmp.txt & 

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_large_patch16 --norm_pix_loss --mask_ratio 0.5 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 9 --arch_version v1 --token_factor 40 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training.csv > tmp.txt & 






## test
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 --use_env main_pretrain.py --batch_size 64 --model mae_vit_base_patch16 --norm_pix_loss --mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --blr 1.5e-4 --weight_decay 0.05 --input_size 224 --num_classes 12 --data_path /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med-SAM2D/ProMISe_Dataset/Datasets/synapseCT/Training/2D_all_5slice/training_full_path.csv 