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

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

def calculate_fid(real_features, fake_features):
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

def calculate_3d_ssim(img1, img2):
    """Calculate SSIM for 3D volumes"""
    if not isinstance(img1, np.ndarray):
        img1 = img1.cpu().numpy()
    if not isinstance(img2, np.ndarray):
        img2 = img2.cpu().numpy()
        
    window_size = min(33, img1.shape[-1]//2)
    if window_size % 2 == 0:
        window_size -= 1
        
    return ssim(img1, img2, 
               data_range=img1.max() - img1.min(),
               win_size=window_size,
               channel_axis=0)

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
    parser.add_argument('--thre', type=float, default=0.1, help='0.1')

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

    save_tensor(x, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_'+class_+'_'+str(args.reverse)+'.png')
    save_tensor(label, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_label_'+class_+'_'+str(args.reverse)+'.png', mask = True)
    save_tensor(image_target, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_target_'+class_+'_'+str(args.reverse)+'.png')

    with torch.no_grad():
        if args.training_version.startswith('v0'):
            middle1 = {"image_target": image_target, "random_selected_class": random_selected_class}
            x_restored, cls_tokens, middle_output = model.forward_encoder(x, mask_ratio=args.mask_ratio, middle=middle1)
            preds = model.forward_decoder(x_restored, cls_tokens, middle_output)
            if args.arch_version.startswith('v1'):
                preds = model.unpatchify(preds)

    # if 0 in args.select_cls: ## without background generation
    #     preds[preds < 0.02] = 0 ## delete noise background by 5 pixel-value

    # # contain [0] is direct gen/seg
    # if 0 in args.select_cls:
    #     image_target = image_target
    #     preds = preds
    # else:
    #     if args.select_cls == []:
    #         image_target = image_target
    #         preds = preds
    #     else: ## indirect gen/seg
    #         image_target = x-image_target
    #         preds = x-preds
    #         preds[preds <= 0] = 0

    preds_thresholded = preds.clone()
    preds_thresholded[preds_thresholded < args.thre] = 0
    image_target_thresholded = image_target.clone()
    image_target_thresholded[image_target_thresholded < args.thre] = 0

    save_tensor(preds, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.reverse)+'.png')
    save_tensor(preds_thresholded, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_pred_image_thresholded_'+str(args.thre)+'_'+class_+'_'+str(args.reverse)+'.png')
    save_tensor(image_target_thresholded, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_target_thresholded_'+str(args.thre)+'_'+class_+'_'+str(args.reverse)+'.png')

    # Calculate metrics for both original and thresholded predictions
    metrics = {}
    
    # Original predictions metrics
    loss_l2 = (preds - image_target) ** 2
    loss_l2 = loss_l2.mean()

    loss_l1 = (preds - image_target)
    loss_l1 = loss_l1.detach().cpu().numpy()
    loss_l1 = np.abs(loss_l1)

    p_loss = torch.mean(perceptual_loss(image_target.contiguous(), preds.contiguous()))
    loss_lpips = p_loss
    loss = (loss_l2 + loss_lpips)

    # Calculate metrics for original predictions
    pred_np = preds[0].detach().cpu().numpy()
    target_np = image_target[0].detach().cpu().numpy()
    
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    ssim_val = ssim(target_np, pred_np, data_range=1.0, channel_axis=0)
    fid_score = calculate_fid(pred_np.reshape(-1, pred_np.shape[-1]), 
                            target_np.reshape(-1, target_np.shape[-1]))

    metrics['original'] = {
        'loss': loss.detach().cpu().numpy(),
        'loss_l1': loss_l1.mean(),
        'loss_l2': loss_l2.detach().cpu().numpy(),
        'loss_lpips': loss_lpips.detach().cpu().numpy(),
        'psnr': psnr_val,
        'ssim': ssim_val,
        'fid': fid_score
    }

    # Calculate metrics for thresholded predictions if applicable
    pred_thresh_np = preds_thresholded[0].detach().cpu().numpy()
    target_thresh_np = image_target_thresholded[0].detach().cpu().numpy()
    
    psnr_val_thresh = psnr(target_thresh_np, pred_thresh_np, data_range=1.0)
    ssim_val_thresh = ssim(target_thresh_np, pred_thresh_np, data_range=1.0, channel_axis=0)
    fid_score_thresh = calculate_fid(pred_thresh_np.reshape(-1, pred_thresh_np.shape[-1]), 
                                   target_thresh_np.reshape(-1, target_thresh_np.shape[-1]))

    loss_l2_thresh = (preds_thresholded - image_target_thresholded) ** 2
    loss_l2_thresh = loss_l2_thresh.mean()

    loss_l1_thresh = np.abs((preds_thresholded - image_target_thresholded).detach().cpu().numpy())

    p_loss_thresh = torch.mean(perceptual_loss(image_target_thresholded.contiguous(), preds_thresholded.contiguous()))
    loss_lpips_thresh = p_loss_thresh

    metrics['thresholded'] = {
        'loss': (loss_l2_thresh + loss_lpips_thresh).detach().cpu().numpy(),
        'loss_l1': loss_l1_thresh.mean(),
        'loss_l2': loss_l2_thresh.detach().cpu().numpy(),
        'loss_lpips': loss_lpips_thresh.detach().cpu().numpy(),
        'psnr': psnr_val_thresh,
        'ssim': ssim_val_thresh,
        'fid': fid_score_thresh
    }

    return metrics


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

    metrics_list = []

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
    
        metrics = gen_one_image(input_img, model_mae, case_id=img, perceptual_loss=perceptual_loss)
        metrics_list.append(metrics)

    print("\nOriginal Predictions Metrics:")
    print("Average Loss:", np.mean([m['original']['loss'] for m in metrics_list]))
    print("Average L1 Loss:", np.mean([m['original']['loss_l1'] for m in metrics_list]))
    print("Average L2 Loss:", np.mean([m['original']['loss_l2'] for m in metrics_list]))
    print("Average LPIPS Loss:", np.mean([m['original']['loss_lpips'] for m in metrics_list]))
    print("Average PSNR:", np.mean([m['original']['psnr'] for m in metrics_list]))
    print("Average SSIM:", np.mean([m['original']['ssim'] for m in metrics_list]))
    print("Average FID:", np.mean([m['original']['fid'] for m in metrics_list]))

    print("\nThresholded Predictions Metrics:")
    print("Average Loss:", np.mean([m['thresholded']['loss'] for m in metrics_list]))
    print("Average L1 Loss:", np.mean([m['thresholded']['loss_l1'] for m in metrics_list]))
    print("Average L2 Loss:", np.mean([m['thresholded']['loss_l2'] for m in metrics_list]))
    print("Average LPIPS Loss:", np.mean([m['thresholded']['loss_lpips'] for m in metrics_list]))
    print("Average PSNR:", np.mean([m['thresholded']['psnr'] for m in metrics_list]))
    print("Average SSIM:", np.mean([m['thresholded']['ssim'] for m in metrics_list]))
    print("Average FID:", np.mean([m['thresholded']['fid'] for m in metrics_list]))
