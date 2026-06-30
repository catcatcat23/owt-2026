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
import OWC2_LIB
import re
from einops import rearrange

from skimage import measure
from skimage.morphology import remove_small_objects, ball, binary_opening, remove_small_holes, binary_closing
import surface_distance
import nibabel as nib

def sorted_nicely( l ): 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def dice_score(prediction, ground_truth):
    assert prediction.shape == ground_truth.shape, "Volumes must have the same shape."

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(prediction, ground_truth).sum()
    size_pred = prediction.sum()
    size_gt = ground_truth.sum()

    if size_pred + size_gt == 0:
        return 1.0

    dice = 2.0 * intersection / (size_pred + size_gt)
    return dice

def calculate_nsd(pred, gt, spacing=(1.5, 1.5, 1.5), tolerance=1.5):
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)
    
    if not np.any(pred) and not np.any(gt):
        return 1.0 #, 0.0
    elif not np.any(pred) or not np.any(gt):
        return 0.0 #, 0.0
        
    surface_distances = surface_distance.compute_surface_distances(gt, pred, spacing)
    nsd = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance)

    return nsd

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

    parser.add_argument('--text_encoding', type=str, default="None", help='None or path of text_encoding')

    return parser

def prepare_model(chkpt_dir, arch, args=None, img_size=None):
    model = OWT_models.__dict__[arch](img_size=img_size, norm_pix_loss=args.norm_pix_loss, model_args=args)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    return model

def filter_class(img, labels, selected_classes):
    image_target = img.clone()
    for ms in selected_classes:
        image_target[labels==ms] = 0
    return image_target

def save_tensor_3D(x, save_name, mask = False, norm=False, save_nii=False):
    # Convert to numpy and permute dimensions
    x_np = x.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy() ## (fr,w,h,c)
    
    # Process for video saving
    x_video = x_np.copy()
    if mask == False:
        if norm == True:
            x_video = (x_video - x_video.min()) / (x_video.max() - x_video.min() + 1e-8)
        x_video = (x_video * 255).astype(np.uint8)
    else:
        x_video = (x_video * 20).astype(np.uint8) ## only for 9 labels

    # Save as video
    x_frames = [x_video[i,:,:,:] for i in range(x_video.shape[0])]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_name+".mp4", fourcc, 20.0, (224, 224))
    for img in x_frames:
        out.write(img)
    out.release()
    
    if save_nii:
        # Save as nii.gz
        # import nibabel as nib
        x_nii = x_np.copy()
        if norm:
            x_nii = (x_nii - x_nii.min()) / (x_nii.max() - x_nii.min() + 1e-8)
        nii_img = nib.Nifti1Image(x_nii, np.eye(4))
        nib.save(nii_img, save_name + '.nii.gz')

def save_tensor_3D_np(x, save_name, mask = False, norm=False, save_nii=False):
    x_video = (x * 255).astype(np.uint8)
    x_video = np.expand_dims(x_video, axis=3)
    h,w,d,_ = x_video.shape
    x_video = np.broadcast_to(x_video, (h, w, d, 3))
    x_np = [x_video[:,:,i,:] for i in range(x_video.shape[2])]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_name+".mp4", fourcc, 20.0, (224, 224))
    for img in x_np:
        out.write(img)
    out.release()
    
    if save_nii:
        x_nii = x.copy()
        if norm:
            x_nii = (x_nii - x_nii.min()) / (x_nii.max() - x_nii.min() + 1e-8)
        nii_img = nib.Nifti1Image(x_nii, np.eye(4))
        nib.save(nii_img, save_name + '.nii.gz')

def gen_one_image(input_img, model, case_id=0, perceptual_loss=None):
    random_selected_class = args.select_cls
    sample1 = input_img[0]
    x = sample1['image'].to(device)
    labels = sample1['label'].to(device)

    image_target = filter_class(x, labels, random_selected_class)

    class_ = 'cls'
    for icl in random_selected_class:
        class_ += str(icl)

    preds = torch.zeros(x.shape).to(device)
    cnts = torch.zeros(x.shape).to(device)

    if args.text_encoding != "None":
        text_features = torch.load(args.text_encoding, map_location="cpu").to(device)
        aligned_text_features = text_features.repeat_interleave(args.token_factor, dim=0)
        aligned_text_features = aligned_text_features.unsqueeze(0).expand(1, -1, -1).float()

    n_inter=0
    total_time = 0.0
    if args.dataset_type == '3D':
        for fr in range(0, x.shape[2]-args.fix_frame+1):
            x_ = x[:,:,fr:fr+args.fix_frame,:,:]
            
            with torch.no_grad():
                middle1 = {"image_target": image_target[:,:,fr:fr+args.fix_frame,:,:], "random_selected_class": random_selected_class}
                if args.text_encoding != "None":
                    middle1["text_features"] = aligned_text_features
                
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                x_restored, cls_tokens, middle_output = model.forward_encoder(x_, mask_ratio=args.mask_ratio, middle=middle1)
                pred1 = model.forward_decoder(x_restored, cls_tokens, middle_output)
                if args.arch_version.startswith('v1'):
                    pred1 = model.unpatchify3D(pred1)
                end_time.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                total_time += elapsed_time
                
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
                
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                x_restored, cls_tokens, middle_output = model.forward_encoder(x_, mask_ratio=args.mask_ratio, middle=middle1)
                pred1 = model.forward_decoder(x_restored, cls_tokens, middle_output)
                if args.arch_version.startswith('v1'):
                    pred1 = model.unpatchify(pred1)
                end_time.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                total_time += elapsed_time
                
                preds[:,:,fr,:,:]+=pred1
                cnts[:,:,fr,:,:]+=1
                n_inter+=1
    
    # Calculate average time per sample (divided by batch size)
    batch_size = x.shape[0]
    avg_time_per_sample = total_time / (n_inter * batch_size)
    print(f"Average processing time per sample: {avg_time_per_sample:.4f} seconds")

    preds = preds/cnts

    if 0 in args.select_cls:
        image_target = image_target
        preds = preds
    else:
        if args.select_cls == []:
            image_target = image_target
            preds = preds
        else:
            image_target = x-image_target
            preds = x-preds
            preds[preds < 0] = 0

    x = x.squeeze().permute(2, 3, 1, 0).detach().cpu().numpy()
    image_target = image_target.squeeze().permute(2, 3, 1, 0).detach().cpu().numpy()
    preds = preds.squeeze().permute(2, 3, 1, 0).detach().cpu().numpy()

    seg_label = np.mean(image_target, axis=3)
    seg_preds = np.mean(preds, axis=3)
    seg_label = (seg_label>args.thre)*1.0
    seg_preds = (seg_preds>args.thre)*1.0

    min_size = 100
    labeled_image = measure.label(seg_preds, connectivity=1)
    cleaned_image = remove_small_objects(labeled_image, min_size=min_size)
    binary_cleaned_image = cleaned_image > 0
    struct_element = ball(1)
    seg_preds = binary_opening(binary_cleaned_image, struct_element)

    dice = dice_score(seg_preds, seg_label)
    nsd_value = calculate_nsd(seg_preds, seg_label)

    if args.save_video == 1:
        save_tensor_3D_np(seg_label, args.output_vis+'/seg_results/'+args.load_csv_type+'/'+case_id+'_seg_label_'+class_+'_'+str(args.thre)+'_'+str(args.reverse)+'.png')
        save_tensor_3D_np(seg_preds, args.output_vis+'/seg_results/'+args.load_csv_type+'/'+case_id+'_seg_preds_'+class_+'_'+str(args.thre)+'_'+str(args.reverse)+'.png')

    return dice, nsd_value

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
    os.makedirs(args.output_vis+'/seg_results/'+args.load_csv_type, exist_ok=True)

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

    img_list = sorted_nicely([i for i in os.listdir(args.load_data_vis_path) if not i.startswith(".")])

    dice_list, nsd_list = [], []

    print("args.save_video", args.save_video)
    print("args.select_cls", args.select_cls)

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
        labels = np.float32(mask)
    
        image = torch.tensor(image)
        labels = torch.tensor(labels)
    
        sample = {'image': image, 'label': labels}
    
        image = sample['image'].unsqueeze(3)
        d, h, w, _ = image.shape
        sample['image'] = image.expand(d, h, w, 3)
    
        labels = sample['label'].unsqueeze(3)
        sample['label'] = labels.expand(d, h, w, 3)   
    
        sample['image'] = sample['image'].permute(3,0,1,2)
        sample['label'] = sample['label'].permute(3,0,1,2)
    
        sample['image'] = sample['image'].unsqueeze(dim=0)
        sample['label'] = sample['label'].unsqueeze(dim=0)
        
        input_img=[]
        input_img.append(sample)
    
        dice, nsd_value = gen_one_image(input_img, model_mae, case_id=img, perceptual_loss=perceptual_loss)

        dice_list.append(dice)
        nsd_list.append(nsd_value)

    print("dice_list", np.mean(dice_list))
    print("nsd_list", np.mean(nsd_list))
