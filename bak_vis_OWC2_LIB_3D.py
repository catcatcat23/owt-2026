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
import OWC2_LIB #models_mae_token2
import re
from einops import rearrange

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

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
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--arch_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--training_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--token_factor', type=int, default=1, help='how many tokens to generate a class')
    parser.add_argument('--loss_version', type=str, default='L2', help='L1-LPIPS-GAN')
    parser.add_argument('--dataset_type', type=str, default='2D', help='2D, 3D') ## but 3D controlled by training_version -3D
    parser.add_argument('--LA', type=bool, default=False, help='False, True')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--select_cls', type=str, default='1', help='select_cls 1,2,3,4,5...')
    parser.add_argument('--reverse', type=int, default=0)
    parser.add_argument('--output_vis', type=str, default=None)


    return parser

def prepare_model(chkpt_dir, arch, args=None, img_size=None):
    # build model
    model = OWC2_LIB.__dict__[arch](img_size=img_size, norm_pix_loss=args.norm_pix_loss, model_args=args)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    return model

# def read_img(img, mask = False):
#     data = cv2.imread(img, cv2.IMREAD_UNCHANGED)
#     data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
#     if mask == False:
#         data = (data-data.min())/(data.max()-data.min()+0.00000001)
#     img = np.float32(data)
#     return img

# def tensor_like(image, device):
#     x = torch.tensor(image)
#     # make it a batch-like
#     x = x.unsqueeze(dim=0)
#     x = torch.einsum('nhwc->nchw', x).to(device, non_blocking=True)
#     return x

def filter_class(img, label, selected_classes):
    image_target = img.clone()
    for ms in selected_classes:
        image_target[label==ms] = 0
    return image_target

def save_tensor(x, save_name, mask = False, norm=False):
    x_np = x.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    if mask == False:
        if norm == True:
            x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        x_np = (x_np * 255).astype(np.uint8)
    else:
        x_np = (x_np * 20).astype(np.uint8) ## only for 9 labels
    cv2.imwrite(save_name, cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR))

def save_tensor_3D(x, save_name, mask = False, norm=False):
    x_np = x.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy() ## (fr,w,h,c)
    if mask == False:
        if norm == True:
            x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        x_np = (x_np * 255).astype(np.uint8)
    else:
        x_np = (x_np * 20).astype(np.uint8) ## only for 9 labels
    for i in range(x_np.shape[0]):
        cv2.imwrite(save_name+"_"+str(i)+".png", cv2.cvtColor(x_np[i,:,:,:], cv2.COLOR_RGB2BGR))

    x_np = [x_np[i,:,:,:] for i in range(x_np.shape[0])]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_name+".mp4", fourcc, 20.0, (224, 224))
    for img in x_np:
        out.write(img)
    out.release()

def gen_one_image(input_img, model, case_id=0):
    device = torch.device(args.device)
    model.to(device)

    # class_list = list(range(1, args.num_classes_with_bg))
    # random.shuffle(class_list)
    # print("class_list", class_list)
    # random_selected_class = class_list[:int(len(class_list)*args.mask_ratio)]

    # random_selected_class = [1]
    random_selected_class = args.select_cls
    print("random_selected_class", random_selected_class)

    # random_selected_class2 = class_list[int(len(class_list)*args.mask_ratio):]
    random_selected_class2 = [0,1,2,3,4,5,6,7,8,9]
    for i in random_selected_class:
        random_selected_class2.remove(i)
    print("random_selected_class2", random_selected_class2)

    assert len(random_selected_class) >= 1
    assert len(random_selected_class2) <= 9

    sample1 = input_img[0]
    x = sample1['image'].to(device, non_blocking=True)
    label = sample1['label'].to(device, non_blocking=True)

    image_target = filter_class(x, label, random_selected_class)
    print("image_target.shape", image_target.shape)

    class_ = 'cls'
    for icl in random_selected_class:
        class_ += str(icl)
    print("class_", class_)
    save_tensor_3D(x, args.output_vis+'/test1_image_'+class_+'_'+str(args.reverse))
    save_tensor_3D(label, args.output_vis+'/test1_label_'+class_+'_'+str(args.reverse), mask = True)
    save_tensor_3D(image_target, args.output_vis+'/test1_image_target_'+class_+'_'+str(args.reverse))

    print("x.shape", x.shape)

    model.eval()

    preds = torch.zeros(x.shape).to(device, non_blocking=True)
    cnts = torch.zeros(x.shape).to(device, non_blocking=True)

    for fr in range(0, x.shape[2]-args.fix_frame):
        x_ = x[:,:,fr:fr+args.fix_frame,:,:]
        print("fr, x_.shape", fr, x_.shape)
    
        if args.training_version.startswith('v0'):
            middle1 = {"image_target": image_target[:,:,fr:fr+args.fix_frame,:,:], "random_selected_class": random_selected_class}
            loss1, pred1, _ = model(x_, mask_ratio=args.mask_ratio, middle=middle1)
            preds[:,:,fr+args.fix_frame,:,:]+=pred1[:,:,-1,:,:]
            cnts[:,:,fr+args.fix_frame,:,:]+=1
        # elif args.training_version.startswith('v1'): ## currently not v1
        #     _random_selected_class = list(range(args.num_classes_with_bg)) ## [0,1,2,3,4,5,6,7,8,9]
        #     for i in random_selected_class:
        #         _random_selected_class.remove(i)
        #     _random_selected_class2 = list(range(args.num_classes_with_bg)) ## [0,1,2,3,4,5,6,7,8,9]
        #     for i in random_selected_class2:
        #         _random_selected_class2.remove(i)
        #     middle1 = {"image_target": x-image_target, "random_selected_class": _random_selected_class}
        #     loss1, pred1, _ = model(image_target, mask_ratio=args.mask_ratio, middle=middle1)
        
    print("loss1", loss1)
    preds = preds/cnts

    save_tensor_3D(preds, args.output_vis+'/test1_pred_image_'+'case'+str(case_id)+'_'+class_+'_'+str(args.reverse)+'.png')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.num_classes_with_bg = args.num_classes + 1
    args.organ_token_total = 1*args.token_factor*1 + args.token_factor*args.num_classes ## 20+180 = 200
    args.organ_token_selet = args.token_factor*int(args.num_classes_with_bg*args.mask_ratio) #len(random_selected_class) ## 100

    args.select_cls = [int(i) for i in args.select_cls.split(',') if i != ''] ## masked in image1, i.e., keep in image2
    print("args.select_cls", args.select_cls)

    # args.if_vq = False
    args.vq_version = None
    args.lib_version = None
    if '-VQ' in args.arch_version:
        # args.if_vq = True
        args.vq_version = args.arch_version.split('-VQ')[1].split('_nt')[0].split('-')[0]
        args.vq_n_token = int(args.arch_version.split('-VQ')[1].split('_nt')[1].split('-')[0])
        if '-LIB' in args.arch_version:
            args.lib_version = args.arch_version.split('-LIB')[1].split('-')[0]

    # args.if_disetg = False
    args.disetg_version = None
    if '-DT' in args.arch_version:
        # args.if_disetg = True
        args.disetg_version = args.arch_version.split('-DT')[1].split('-')[0]

    args.cls_num = 1
    if '-cls' in args.arch_version:
        args.cls_num = int(args.arch_version.split('-cls')[1].split('-')[0])

    args.arch_version = args.arch_version.split('-')[0]

    args.fix_frame = 0
    if '-3D' in args.training_version:
        args.dataset_type = '3D'
        if '-Fixfr' in args.training_version:
            args.fix_frame = int(args.training_version.split("-Fixfr")[1])
        args.training_version = args.training_version.split("-3D")[0]

    if '-LA' in args.model:
        args.LA = True
        args.model = args.model.split("-LA")[0]
    print(args.model)

    # path = '/mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/'
    path = '/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/'
    # img1 = path+'Training/image/BDMAP_00000001/BDMAP_00000001_194.jpg'
    # label1 = path+'Training/mask/BDMAP_00000001/BDMAP_00000001_194.png'
    img2 = path+'Test/image/BDMAP_00000068/'
    label2 = path+'Test/mask/BDMAP_00000068/'

    # img2 = path+'Test/image/BDMAP_00000031/BDMAP_00000031_309.jpg'
    # label2 = path+'Test/mask/BDMAP_00000031/BDMAP_00000031_309.png'
    img1 = path+'Test/image/BDMAP_00000055/'
    label1 = path+'Test/mask/BDMAP_00000055/'
    
    if args.reverse == 1:
        img1 = path+'Training/image/BDMAP_00000030/'
        label1 = path+'Training/mask/BDMAP_00000030/'

        img2 = path+'Test/image/BDMAP_00000055/'
        label2 = path+'Test/mask/BDMAP_00000055/'

    # image1 = read_img(img1)
    # label1 = read_img(label1, mask = True)
    # sample1 = {'image': image1, 'label':label1}
    
    # image2 = read_img(img2)
    # label2 = read_img(label2, mask = True)
    # sample2 = {'image': image2, 'label':label2}
    
    data_path = img1
    samp_list = sorted_nicely([i_s for i_s in os.listdir(data_path) if not i_s.startswith(".")])
    images = [cv2.imread(img1+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in samp_list]
    image = np.stack(images)
    # image = np.transpose(image, (1, 2, 0))
    image = np.float32(image)
    image = (image-image.min())/(image.max()-image.min()+0.00000001)
    # image = image[40:40+args.fix_frame]

    mask_path = label1
    mask_list = sorted_nicely([i_s for i_s in os.listdir(mask_path) if not i_s.startswith(".")])
    masks = [cv2.imread(mask_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in mask_list]
    mask = np.stack(masks)
    # mask = np.transpose(mask, (1, 2, 0))
    label = np.float32(mask)
    # label = label[40:40+args.fix_frame]

    image = torch.tensor(image)
    label = torch.tensor(label)

    sample = {'image': image, 'label': label}

    image = sample['image'].unsqueeze(3)
    d, h, w, _ = image.shape
    sample['image'] = image.expand(d, h, w, 3)

    label = sample['label'].unsqueeze(3)
    sample['label'] = label.expand(d, h, w, 3)   

    print("sample['image'].shape 2", sample['image'].shape)
    print("sample['label'].shape 2", sample['label'].shape)

    # if d % 16 != 0:
    #     pad_size = 16 - (d % 16)
    #     print("pad_size", pad_size)
    #     pad_tensor = torch.zeros(pad_size, h, w, 3, dtype=sample['image'].dtype)
    #     pad_tensor2 = torch.zeros(pad_size, h, w, 3, dtype=sample['label'].dtype)
    #     sample['image'] = torch.cat((sample['image'], pad_tensor), dim=0)
    #     sample['label'] = torch.cat((sample['label'], pad_tensor2), dim=0)
    sample['image'] = sample['image'].permute(3,0,1,2)
    sample['label'] = sample['label'].permute(3,0,1,2)

    sample['image'] = sample['image'].unsqueeze(dim=0)
    sample['label'] = sample['label'].unsqueeze(dim=0)
    print("sample['image'].shape 3", sample['image'].shape)
    print("sample['label'].shape 3", sample['label'].shape)
    print("torch.unique(sample['label'])", torch.unique(sample['label']))

    # print("image1.shape, image2.shape", image1.shape, image2.shape)
    
    input_img=[]
    input_img.append(sample)
    # input_img.append(sample2)
    
    chkpt_dir = args.checkpoint #'/mnt/weka/wekafs/rad-megtron/ss3112/Github/mae/Results/bak/bak3_output_dir_withbg/checkpoint-560.pth'
    model_mae = prepare_model(chkpt_dir, args.model, args, img_size=args.input_size)
    
    print('Model loaded.')
    
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)

    gen_one_image(input_img, model_mae, case_id=0)




