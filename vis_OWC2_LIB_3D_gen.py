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
import OWT_models
import re
from einops import rearrange

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

def calculate_3d_ssim(img1, img2):
    if not isinstance(img1, np.ndarray):
        img1 = img1.cpu().numpy()
    if not isinstance(img2, np.ndarray):
        img2 = img2.cpu().numpy()
        
    window_size = min(33, img1.shape[-1]//2)
    if window_size % 2 == 0:
        window_size -= 1
        
    result = ssim(img1, img2, 
               data_range=img1.max() - img1.min(),
               win_size=window_size,
               channel_axis=0)
               
    if np.isnan(result):
        return 0.0
    return result

def sorted_nicely( l ): 
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

    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

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

    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--arch_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--training_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--token_factor', type=int, default=20, help='how many tokens to generate a class')
    parser.add_argument('--loss_version', type=str, default='L2', help='L1-LPIPS-GAN')
    parser.add_argument('--dataset_type', type=str, default='2D', help='2D, 3D')
    parser.add_argument('--LA', type=bool, default=False, help='False, True')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--select_cls', type=str, default='1', help='select_cls 1,2,3,4,5...')
    parser.add_argument('--reverse', type=int, default=0)
    parser.add_argument('--output_vis', type=str, default=None)

    parser.add_argument('--load_data_vis_path', type=str, default=None)
    parser.add_argument('--load_label_vis_path', type=str, default=None)
    parser.add_argument('--load_csv_type', type=str, default='train')
    parser.add_argument('--save_video', type=int, default=0, help='0=False, 1=True')
    parser.add_argument('--thre', type=float, default=0.1, help='0.1')
    parser.add_argument('--id_index', type=int, default=0, help='id_index')
    parser.add_argument('--save_only', action='store_true',
                        help='save visualizations without LPIPS/SSIM metrics')

    parser.add_argument('--text_encoding', type=str, default="None", help='None or path of text_encoding')

    return parser

def prepare_model(chkpt_dir, arch, args=None, img_size=None):
    import OWT_models
    model = OWT_models.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss, model_args=args)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    state = checkpoint['model']
    if args.save_only:
        state = {
            key: value for key, value in state.items()
            if not key.startswith('perceptual_loss.')
        }
    msg = model.load_state_dict(state, strict=True)
    print(msg)
    return model

def filter_class(img, label, selected_classes):
    image_target = img.clone()
    for ms in selected_classes:
        image_target[label==ms] = 0
    return image_target

def save_tensor_3D(x, save_name, mask = False, norm=False, save_video = 1):
    x_np = x.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy()
    if mask == False:
        if norm == True:
            x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        x_np = (x_np * 255).astype(np.uint8)
    else:
        x_np = (x_np * 20).astype(np.uint8)

    if save_video == 2:
        for i in range(x_np.shape[0]):
            cv2.imwrite(save_name+"_"+str(i)+".png", cv2.cvtColor(x_np[i,:,:,:], cv2.COLOR_RGB2BGR))

    x_np = [x_np[i,:,:,:] for i in range(x_np.shape[0])]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_name+".mp4", fourcc, 20.0, (224, 224))
    for img in x_np:
        out.write(img)
    out.release()

def gen_one_image(input_img, model, case_id=0, perceptual_loss=None):
    random_selected_class = args.select_cls
    sample1 = input_img[0]
    x = sample1['image'].to(device)
    label = sample1['label'].to(device)

    image_target = filter_class(x, label, random_selected_class)
    class_ = 'cls'
    for icl in random_selected_class:
        class_ += str(icl)

    if args.save_video == 1 or args.save_video == 2:
        save_tensor_3D(x, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_'+class_+'_'+str(args.reverse), save_video = args.save_video)
        save_tensor_3D(label, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_label_'+class_+'_'+str(args.reverse), mask = True, save_video = args.save_video)
        save_tensor_3D(image_target, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_target_'+class_+'_'+str(args.reverse), save_video = args.save_video)

    preds = torch.zeros(x.shape).to(device)
    cnts = torch.zeros(x.shape).to(device)

    if args.text_encoding != "None":
        text_features = torch.load(args.text_encoding, map_location="cpu").to(device)
        aligned_text_features = text_features.repeat_interleave(args.token_factor, dim=0)
        aligned_text_features = aligned_text_features.unsqueeze(0).expand(1, -1, -1).float()

    n_inter=0
    if args.dataset_type == '3D':
        for fr in range(0, x.shape[2]-args.fix_frame+1):
            x_ = x[:,:,fr:fr+args.fix_frame,:,:]
            with torch.no_grad():
                middle1 = {"image_target": image_target[:,:,fr:fr+args.fix_frame,:,:], "random_selected_class": random_selected_class}
                if args.text_encoding != "None":
                    middle1["text_features"] = aligned_text_features
                x_restored, cls_tokens, middle_output = model.forward_encoder(x_, mask_ratio=args.mask_ratio, middle=middle1)
                pred1 = model.forward_decoder(x_restored, cls_tokens, middle_output)
                if args.arch_version.startswith('v1'):
                    pred1 = model.unpatchify3D(pred1)
                preds[:,:,fr:fr+args.fix_frame,:,:]+=pred1
                cnts[:,:,fr:fr+args.fix_frame,:,:]+=1
                n_inter+=1
    elif args.dataset_type == '2D':
        for fr in range(0, x.shape[2]):
            x_ = x[:,:,fr,:,:]
            with torch.no_grad():
                middle1 = {"image_target": image_target[:,:,fr,:,:], "random_selected_class": random_selected_class}
                if args.text_encoding != "None":
                    middle1["text_features"] = aligned_text_features
                x_restored, cls_tokens, middle_output = model.forward_encoder(x_, mask_ratio=args.mask_ratio, middle=middle1)
                pred1 = model.forward_decoder(x_restored, cls_tokens, middle_output)
                if args.arch_version.startswith('v1'):
                    pred1 = model.unpatchify(pred1)
                preds[:,:,fr,:,:]+=pred1
                cnts[:,:,fr,:,:]+=1
                n_inter+=1

    preds = preds/cnts
    preds[preds < 0] = 0

    preds_thresholded = preds.clone()
    preds_thresholded[preds_thresholded < args.thre] = 0
    image_target_thresholded = image_target.clone()
    image_target_thresholded[image_target_thresholded < args.thre] = 0
    
    if args.save_video == 1 or args.save_video == 2:
        save_tensor_3D(preds, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.reverse)+'.png', save_video = args.save_video)
        save_tensor_3D(preds_thresholded, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_pred_image_thresholded_'+str(args.thre)+'_'+class_+'_'+str(args.reverse)+'.png', save_video = args.save_video)
        save_tensor_3D(image_target_thresholded, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_target_thresholded_'+str(args.thre)+'_'+class_+'_'+str(args.reverse)+'.png', save_video = args.save_video)

    if args.save_only:
        return None

    metrics = {}
    
    loss_l2 = (preds - image_target) ** 2
    loss_l2 = loss_l2.mean()

    loss_l1 = (preds - image_target)
    loss_l1 = loss_l1.detach().cpu().numpy()
    loss_l1 = np.abs(loss_l1)

    p_loss = 0
    loss_count = 0
    for i_sl in range(image_target.shape[2]):
        image_target_i = image_target[:,:,i_sl,:,:]
        pred_i = preds[:,:,i_sl,:,:]
        p_loss = p_loss + torch.mean(perceptual_loss(image_target_i.contiguous(), pred_i.contiguous()))
        loss_count+=1
    loss_lpips = p_loss/loss_count

    ssim_scores = []
    for i in range(preds.shape[2]):
        pred_slice = preds[0,:,i,:,:].detach().cpu().numpy()
        target_slice = image_target[0,:,i,:,:].detach().cpu().numpy()
        ssim_val = ssim(target_slice, pred_slice, data_range=1.0, channel_axis=0)
        ssim_scores.append(ssim_val)

    ssim_3d = calculate_3d_ssim(preds[0].detach(), image_target[0].detach())

    metrics['original'] = {
        'loss_l1': loss_l1.mean(),
        'loss_l2': loss_l2.detach().cpu().numpy(),
        'loss_lpips': loss_lpips.detach().cpu().numpy(),
        'ssim_avg': np.mean(ssim_scores),
        'ssim_3d': ssim_3d
    }

    loss_l2_thresh = (preds_thresholded - image_target_thresholded) ** 2
    loss_l2_thresh = loss_l2_thresh.mean()

    loss_l1_thresh = np.abs((preds_thresholded - image_target_thresholded).detach().cpu().numpy())

    p_loss_thresh = 0
    for i_sl in range(image_target_thresholded.shape[2]):
        image_target_i = image_target_thresholded[:,:,i_sl,:,:]
        pred_i = preds_thresholded[:,:,i_sl,:,:]
        p_loss_thresh = p_loss_thresh + torch.mean(perceptual_loss(image_target_i.contiguous(), pred_i.contiguous()))
    loss_lpips_thresh = p_loss_thresh/loss_count

    ssim_scores_thresh = []
    for i in range(preds_thresholded.shape[2]):
        pred_slice = preds_thresholded[0,:,i,:,:].detach().cpu().numpy()
        target_slice = image_target_thresholded[0,:,i,:,:].detach().cpu().numpy()
        ssim_val = ssim(target_slice, pred_slice, data_range=1.0, channel_axis=0)
        ssim_scores_thresh.append(ssim_val)

    ssim_3d_thresh = calculate_3d_ssim(preds_thresholded[0].detach(), image_target_thresholded[0].detach())

    metrics['thresholded'] = {
        'loss_l1': loss_l1_thresh.mean(),
        'loss_l2': loss_l2_thresh.detach().cpu().numpy(),
        'loss_lpips': loss_lpips_thresh.detach().cpu().numpy(),
        'ssim_avg': np.mean(ssim_scores_thresh),
        'ssim_3d': ssim_3d_thresh
    }

    return metrics

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.num_classes_with_bg = args.num_classes + 1
    args.organ_token_total = 1*args.token_factor*1 + args.token_factor*args.num_classes
    args.organ_token_selet = args.token_factor*int(args.num_classes_with_bg*args.mask_ratio)

    args.select_cls = [int(i) for i in args.select_cls.split(',') if i != '']
    print("args.select_cls", args.select_cls)

    args.cls_num = 1
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
    perceptual_loss = None
    if not args.save_only:
        from VQ.lpips import LPIPS
        perceptual_loss = LPIPS().to(device).eval()
    
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)

    img_list = sorted_nicely([i for i in os.listdir(args.load_data_vis_path) if not i.startswith(".")])

    metrics_list = []

    print("args.save_video", args.save_video)
    print("args.select_cls", args.select_cls)

    break_id = 0

    for img in img_list:
        data_path = args.load_data_vis_path+'/'+img
        samp_list = sorted_nicely([i_s for i_s in os.listdir(data_path) if not i_s.startswith(".")])
        images = [cv2.imread(data_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in samp_list]
        image = np.stack(images)
        image = np.float32(image)
        image = (image-image.min())/(image.max()-image.min()+0.00000001)
    
        mask_path = args.load_label_vis_path+'/'+img
        mask_list = sorted_nicely([i_s for i_s in os.listdir(mask_path) if not i_s.startswith(".")])
        masks = [cv2.imread(mask_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in mask_list]
        mask = np.stack(masks)
        label = np.float32(mask)
    
        image = torch.tensor(image)
        label = torch.tensor(label)
    
        sample = {'image': image, 'label': label}
    
        image = sample['image'].unsqueeze(3)
        d, h, w, _ = image.shape
        sample['image'] = image.expand(d, h, w, 3)
    
        label = sample['label'].unsqueeze(3)
        sample['label'] = label.expand(d, h, w, 3)   
    
        sample['image'] = sample['image'].permute(3,0,1,2)
        sample['label'] = sample['label'].permute(3,0,1,2)
    
        sample['image'] = sample['image'].unsqueeze(dim=0)
        sample['label'] = sample['label'].unsqueeze(dim=0)
        
        input_img=[]
        input_img.append(sample)
    
        metrics = gen_one_image(input_img, model_mae, case_id=img, perceptual_loss=perceptual_loss)
        if metrics is not None:
            metrics_list.append(metrics)

        break_id +=1
        if break_id == args.id_index:
            break

    if args.save_only:
        print("Saved visualization outputs without reconstruction metrics.")
        sys.exit(0)

    print("\nOriginal Predictions Metrics:")
    print("Average L1 Loss:", np.mean([m['original']['loss_l1'] for m in metrics_list]))
    print("Average L2 Loss:", np.mean([m['original']['loss_l2'] for m in metrics_list]))
    print("Average LPIPS Loss:", np.mean([m['original']['loss_lpips'] for m in metrics_list]))
    print("Average SSIM:", np.mean([m['original']['ssim_avg'] for m in metrics_list]))
    print("Average 3D SSIM:", np.mean([m['original']['ssim_3d'] for m in metrics_list]))

    print("\nThresholded Predictions Metrics:")
    print("Average L1 Loss:", np.mean([m['thresholded']['loss_l1'] for m in metrics_list]))
    print("Average L2 Loss:", np.mean([m['thresholded']['loss_l2'] for m in metrics_list]))
    print("Average LPIPS Loss:", np.mean([m['thresholded']['loss_lpips'] for m in metrics_list]))
    print("Average SSIM:", np.mean([m['thresholded']['ssim_avg'] for m in metrics_list]))
    print("Average 3D SSIM:", np.mean([m['thresholded']['ssim_3d'] for m in metrics_list]))
