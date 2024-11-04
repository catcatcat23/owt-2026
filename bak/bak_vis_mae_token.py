import sys
import os
import requests
import cv2
import argparse

import torch
import numpy as np
import random

import matplotlib.pyplot as plt
from PIL import Image
import pickle
from scipy.ndimage import zoom

# check whether run in Colab
# if 'google.colab' in sys.modules:
#     print('Running in Colab.')
#     !pip3 install timm==0.4.5  # 0.3.2 does not work in Colab
#     !git clone https://github.com/facebookresearch/mae.git
#     sys.path.append('./mae')
# else:
sys.path.append('..')
import models_mae_token2

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')

    ## new
    # parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--arch_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--token_factor', type=int, default=1, help='how many tokens to generate a class')


    return parser

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', args=None):
    # build model
    # model = getattr(models_mae, arch)()
    # model = getattr(models_mae_token2, arch)()
    model = models_mae_token2.__dict__[arch](norm_pix_loss=args.norm_pix_loss, model_args=args)
    # load model
    # checkpoint = torch.load(chkpt_dir, map_location='cpu')
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    return model

def gen_one_image(input_img, model):
    device = torch.device(args.device)
    model.to(device)

    class_list = list(range(1, args.num_classes_with_bg))
    random.shuffle(class_list)
    print("class_list", class_list)
    random_selected_class = class_list[:int(len(class_list)*args.mask_ratio)]
    print("random_selected_class", random_selected_class)

    random_selected_class2 = class_list[int(len(class_list)*args.mask_ratio):]
    print("random_selected_class2", random_selected_class2)


    sample1, sample2 = input_img[0], input_img[1]
    # image1 = input_img[0]
    # image2 = input_img[1]
    image1 = sample1['image']
    label = sample1['label']

    image2 = sample2['image']

    x = torch.tensor(image1)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x1 = torch.einsum('nhwc->nchw', x).to(device, non_blocking=True)

    label = torch.tensor(label)
    # make it a batch-like
    label = label.unsqueeze(dim=0)
    label = torch.einsum('nhwc->nchw', label).to(device, non_blocking=True)

    image_target = x1.clone()
    for ms in random_selected_class:
        image_target[label==ms] = 0

    image_target_np = image_target.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image_target_np = (image_target_np - image_target_np.min()) / (image_target_np.max() - image_target_np.min() + 1e-8)
    image_target_np = (image_target_np * 255).astype(np.uint8)
    image_target_path = 'tmp/test_image_target.png'
    cv2.imwrite(image_target_path, cv2.cvtColor(image_target_np, cv2.COLOR_RGB2BGR))

    image_target_np = label.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image_target_np = (image_target_np * 20).astype(np.uint8)
    image_target_path = 'tmp/test_label.png'
    cv2.imwrite(image_target_path, cv2.cvtColor(image_target_np, cv2.COLOR_RGB2BGR))

    image_target_np = x1.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image_target_np = (image_target_np - image_target_np.min()) / (image_target_np.max() - image_target_np.min() + 1e-8)
    image_target_np = (image_target_np * 255).astype(np.uint8)
    image_target_path = 'tmp/test_image.png'
    cv2.imwrite(image_target_path, cv2.cvtColor(image_target_np, cv2.COLOR_RGB2BGR))

    x = torch.tensor(image2)
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x2 = torch.einsum('nhwc->nchw', x).to(device, non_blocking=True)

    print("x1.shape", x1.shape)
    print("x2.shape", x2.shape)

    middle1 = {"image_target": image_target, "random_selected_class": random_selected_class}
    middle2 = {"image_target": x2, "random_selected_class": random_selected_class2}

    model.eval()
    loss1, pred1, _ = model(x1, mask_ratio=args.mask_ratio, middle=middle1)
    # pred1 = model.unpatchify(pred1)
    print("loss1", loss1)
    print("pred1.shape", pred1.shape)

    pred1_image = pred1.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    print("pred1", np.unique(pred1_image))
    pred1_image = (pred1_image * 255).astype(np.uint8)
    # print("pred1", np.unique(pred1_image))
    pred1_image_path = 'tmp/test_pred1_image.png'
    cv2.imwrite(pred1_image_path, cv2.cvtColor(pred1_image, cv2.COLOR_RGB2BGR))
    print(f"Prediction saved as {pred1_image_path}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.num_classes_with_bg = args.num_classes + 1
    args.organ_token_total = 1*args.token_factor*1 + args.token_factor*args.num_classes ## 20+180 = 200
    args.organ_token_selet = args.token_factor*int(args.num_classes_with_bg*args.mask_ratio) #len(random_selected_class) ## 100

    path = '/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/'
    # img1 = path+'Test/image/BDMAP_00000055/BDMAP_00000055_78.jpg'
    # label1 = path+'Test/mask/BDMAP_00000055/BDMAP_00000055_78.png'
    # img1 = path+'Training/image/BDMAP_00000001/BDMAP_00000001_194.jpg'
    # label1 = path+'Training/mask/BDMAP_00000001/BDMAP_00000001_194.png'
    img1 = path+'Test/image/BDMAP_00000068/BDMAP_00000068_81.jpg'
    label1 = path+'Test/mask/BDMAP_00000068/BDMAP_00000068_81.png'
    img2 = path+'Test/image/BDMAP_00000068/BDMAP_00000068_81.jpg'
    
    data1 = cv2.imread(img1, cv2.IMREAD_UNCHANGED)
    data1 = cv2.cvtColor(data1, cv2.COLOR_BGR2RGB)
    data1 = (data1-data1.min())/(data1.max()-data1.min()+0.00000001)
    image1 = np.float32(data1)
    mask = cv2.imread(label1, cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    label1 = np.float32(mask)
    sample1 = {'image': image1, 'label':label1}
    
    data2 = cv2.imread(img2, cv2.IMREAD_UNCHANGED)
    data2 = cv2.cvtColor(data2, cv2.COLOR_BGR2RGB)
    data2 = (data2-data2.min())/(data2.max()-data2.min()+0.00000001)
    image2 = np.float32(data2)
    sample2 = {'image': image2}
    
    print("image1.shape, image2.shape", image1.shape, image2.shape)
    
    input_img=[]
    input_img.append(sample1)
    input_img.append(sample2)
    
    chkpt_dir = '/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/output_dir/checkpoint-420.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16', args)
    
    print('Model loaded.')
    
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    gen_one_image(input_img, model_mae)




