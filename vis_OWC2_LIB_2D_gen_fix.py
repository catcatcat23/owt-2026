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

    ## new
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--arch_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--training_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--token_factor', type=int, default=20, help='how many tokens to generate a class')
    parser.add_argument('--loss_version', type=str, default='L2', help='L1-LPIPS-GAN')
    parser.add_argument('--dataset_type', type=str, default='2D', help='2D, 3D') ## but 3D controlled by training_version -3D
    parser.add_argument('--LA', type=bool, default=False, help='False, True')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--select_cls', type=str, default='1', help='select_cls 1,2,3,4,5...')
    parser.add_argument('--reverse', type=int, default=0)
    parser.add_argument('--output_vis', type=str, default=None)

    parser.add_argument('--load_data_vis_path', type=str, default=None)
    parser.add_argument('--load_label_vis_path', type=str, default=None)
    parser.add_argument('--load_csv_type', type=str, default='train') ## train, test
    parser.add_argument('--save_video', type=int, default=0, help='0=False, 1=True')

    parser.add_argument('--text_encoding', type=str, default="None", help='None or path of text_encoding')

    return parser

def prepare_model(chkpt_dir, arch, args=None, img_size=None):
    # build model
    model = OWC2_LIB.__dict__[arch](img_size=img_size, norm_pix_loss=args.norm_pix_loss, model_args=args)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    return model

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

def gen_one_image(input_img, model, case_id=0, perceptual_loss=None):
    random_selected_class = args.select_cls
    sample1 = input_img[0]
    x = sample1['image'].to(device)
    label = sample1['label'].to(device)

    image_target = filter_class(x, label, random_selected_class)

    class_ = 'cls'
    for icl in random_selected_class:
        class_ += str(icl)

    if args.save_video == 1:
        save_tensor(x, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_'+class_+'_'+str(args.reverse)+'.png')
        save_tensor(label, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_label_'+class_+'_'+str(args.reverse)+'.png', mask = True)
        save_tensor(image_target, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_target_'+class_+'_'+str(args.reverse)+'.png')

    inter_feature = torch.zeros((1, int(args.token_factor*args.num_classes_with_bg), 768)).to(device)
    inter_feature_2 = torch.zeros((1, int(args.token_factor*args.num_classes_with_bg), 768)).to(device)

    with torch.no_grad():
        if args.training_version.startswith('v0'):
            middle1 = {"image_target": image_target, "random_selected_class": random_selected_class}
            x_restored, cls_tokens, middle_output = model.forward_encoder(x, mask_ratio=args.mask_ratio, middle=middle1)
            pred1 = model.forward_decoder(x_restored, cls_tokens, middle_output)
            if args.arch_version.startswith('v1'):
                pred1 = model.unpatchify(pred1)
            preds = pred1

            if args.select_cls == []:
                inter_feature[0,:,:] = middle_output['x_masked_b'] #x_restored
                inter_feature_2[0,:,:] = x_restored #x_restored

    if args.save_video == 1:
        save_tensor(preds, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.reverse)+'.png')

    if args.select_cls == []:
        inter_feature = inter_feature.detach().cpu().numpy()
        inter_feature_2 = inter_feature_2.detach().cpu().numpy()
        if args.save_video == 1:
            np.save(args.output_vis+'/inter_features/'+args.load_csv_type+'/'+case_id+'.npy', inter_feature)
            np.save(args.output_vis+'/inter_features/'+args.load_csv_type+'/'+case_id+'_2.npy', inter_feature_2)

    loss_l2 = (preds - image_target) ** 2
    loss_l2 = loss_l2.mean()

    loss_l1 = (preds - image_target)
    loss_l1 = loss_l1.detach().cpu().numpy()
    loss_l1 = np.abs(loss_l1)

    loss_lpips = torch.mean(perceptual_loss(image_target.contiguous(), preds.contiguous()))
    loss = (loss_l2 + loss_lpips)

    return loss.detach().cpu().numpy(), loss_l1.mean(), loss_l2.detach().cpu().numpy(), loss_lpips.detach().cpu().numpy()

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
    args.temp_stride = 0
    if '-3D' in args.training_version:
        args.dataset_type = '3D'
        if '-Fixfr' in args.training_version:
            args.fix_frame = int(args.training_version.split("-Fixfr")[1].split("-")[0])
        if '-TS' in args.training_version:
            args.temp_stride = int(args.training_version.split("-TS")[1].split("-")[0])
        args.training_version = args.training_version.split("-3D")[0]

    if '-LA' in args.model:
        args.LA = True
        args.model = args.model.split("-LA")[0]
    print(args.model)

    os.makedirs(args.output_vis+'/masked_results/'+args.load_csv_type, exist_ok=True)
    os.makedirs(args.output_vis+'/inter_features/'+args.load_csv_type, exist_ok=True)

    chkpt_dir = args.checkpoint
    model_mae = prepare_model(chkpt_dir, args.model, args, img_size=args.input_size)

    device = torch.device(args.device)
    model_mae.to(device)
    model_mae.eval()
    
    print('Model loaded.')
    from VQ.lpips import LPIPS
    perceptual_loss = LPIPS().to(device).eval()
    
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)

    ## load img
    img_list = sorted_nicely([i for i in os.listdir(args.load_data_vis_path) if not i.startswith(".")])
    # print(img_list)

    loss_list = []
    loss_l1_list = []
    loss_l2_list = []
    loss_lpips_list = []

    print("args.save_video", args.save_video)
    print("args.select_cls", args.select_cls)

    for img in img_list:
        data_path = args.load_data_vis_path+'/'+img
        image = cv2.imread(data_path)  # Read as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = np.float32(image)
        image = (image-image.min())/(image.max()-image.min()+0.00000001)
    
        mask_path = args.load_label_vis_path+'/'+img.split(".jpg")[0]+".png"
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        label = np.float32(mask)
    
        image = torch.tensor(image)
        label = torch.tensor(label)
    
        sample = {'image': image, 'label': label}
    
        # label = sample['label'].unsqueeze(2)
        # sample['label'] = label.expand(label.shape[0], label.shape[1], 3)   
    
        sample['image'] = sample['image'].permute(2,0,1)
        sample['label'] = sample['label'].permute(2,0,1)
    
        sample['image'] = sample['image'].unsqueeze(dim=0)
        sample['label'] = sample['label'].unsqueeze(dim=0)
        
        input_img=[]
        input_img.append(sample)
    
        loss, loss_l1, loss_l2, loss_lpips = gen_one_image(input_img, model_mae, case_id=img.split('.')[0], perceptual_loss=perceptual_loss)
        loss_list.append(loss)
        loss_l1_list.append(loss_l1)
        loss_l2_list.append(loss_l2)
        loss_lpips_list.append(loss_lpips)

    print("loss_avg", np.mean(loss_list))
    print("loss_l1_avg", np.mean(loss_l1_list))
    print("loss_l2_avg", np.mean(loss_l2_list))
    print("loss_lpips_avg", np.mean(loss_lpips_list))



















