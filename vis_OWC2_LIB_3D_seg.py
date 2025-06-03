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

# from scipy.ndimage import binary_erosion, distance_transform_edt
# from skimage.filters import threshold_otsu
from skimage import measure
from skimage.morphology import remove_small_objects, ball, binary_opening, remove_small_holes, binary_closing
import surface_distance
# from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
import nibabel as nib

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
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
        # Avoid division by zero
        return 1.0

    dice = 2.0 * intersection / (size_pred + size_gt)
    return dice

# def surface_voxels(binary_mask):
#     structure = np.ones((3, 3, 3), dtype=bool)
#     eroded = binary_erosion(binary_mask, structure=structure, iterations=1)
#     surface = binary_mask ^ eroded
#     return surface

# def nsd(prediction, ground_truth, voxel_spacing=(1,1,1), tolerance=1):
#     assert prediction.shape == ground_truth.shape, "Volumes must have the same shape."
#     prediction = prediction.astype(bool)
#     ground_truth = ground_truth.astype(bool)

#     # Extract surfaces
#     surface_pred = surface_voxels(prediction)
#     surface_gt = surface_voxels(ground_truth)

#     # Compute distance maps
#     dt_gt = distance_transform_edt(~ground_truth, sampling=voxel_spacing)
#     dt_pred = distance_transform_edt(~prediction, sampling=voxel_spacing)

#     # Surface distances
#     sds_pred = dt_gt[surface_pred]
#     sds_gt = dt_pred[surface_gt]

#     num_pred_within_tol = np.sum(sds_pred <= tolerance)
#     num_gt_within_tol = np.sum(sds_gt <= tolerance)

#     num_surface_voxels = surface_pred.sum() + surface_gt.sum()

#     if num_surface_voxels == 0:
#         return 1.0  # Both surfaces are empty

#     nsd_value = (num_pred_within_tol + num_gt_within_tol) / num_surface_voxels
#     return nsd_value

def calculate_nsd(pred, gt, spacing=(1.5, 1.5, 1.5), tolerance=1.5):
    """
    Calculate Normalized Surface Dice between prediction and ground truth
    
    Args:
        pred: Binary prediction array
        gt: Binary ground truth array
        spacing: Voxel spacing in mm (default: isotropic 1mm)
        tolerance: Distance tolerance in mm
        
    Returns:
        nsd: Normalized Surface Dice score
    """
    # Ensure binary masks
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)
    
    # If either mask is empty, handle edge cases
    if not np.any(pred) and not np.any(gt):
        return 1.0
    elif not np.any(pred) or not np.any(gt):
        return 0.0
        
    # Compute surface distances
    surface_distances = surface_distance.compute_surface_distances(gt, pred, spacing)
    
    # Calculate normalized surface dice
    nsd = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance)

    expected_average_surface_distance=(surface_distance.compute_average_surface_distance(surface_distances))
    expected_hausdorff_100=(surface_distance.compute_robust_hausdorff(surface_distances, 100))
    expected_hausdorff_95=surface_distance.compute_robust_hausdorff(surface_distances, 95)
    expected_surface_overlap=(surface_distance.compute_surface_overlap_at_tolerance(surface_distances, tolerance)),
    # expected_volumetric_dice=surface_distance.compute_dice_coefficient(pred, gt)
    # print("expected_volumetric_dice", expected_volumetric_dice) ## same to other dice

    return nsd, expected_average_surface_distance, expected_hausdorff_100, expected_hausdorff_95, expected_surface_overlap

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

def filter_class(img, labels, selected_classes):
    image_target = img.clone()
    for ms in selected_classes:
        image_target[labels==ms] = 0
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
    # Save as video
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
        # Save as nii.gz
        # import nibabel as nib
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

    ## segmentation evaluation
    # assert len(random_selected_class) == 1

    class_ = 'cls'
    for icl in random_selected_class:
        class_ += str(icl)
    # print("class_", class_)
    if args.save_video == 1:
        save_tensor_3D(x, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_'+class_+'_'+str(args.reverse))
        save_tensor_3D(labels, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_label_'+class_+'_'+str(args.reverse), mask = True)
        save_tensor_3D(image_target, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_image_target_'+class_+'_'+str(args.reverse))

    preds = torch.zeros(x.shape).to(device)
    cnts = torch.zeros(x.shape).to(device)
    # inter_feature = torch.zeros((int(112/16), int(args.token_factor*args.num_classes_with_bg), 768)).to(device)

    if args.text_encoding != "None":
        # print("args.text_encoding", args.text_encoding, os.path.exists(args.text_encoding))
        text_features = torch.load(args.text_encoding, map_location="cpu").to(device)
        aligned_text_features = text_features.repeat_interleave(args.token_factor, dim=0)  # Shape (100, 512)
        aligned_text_features = aligned_text_features.unsqueeze(0).expand(1, -1, -1).float()  # Shape (B, 100, 512) ## in eval (1, 100, 512)

    n_inter=0
    if args.dataset_type == '3D':
        # for fr in range(0, x.shape[2]-args.fix_frame+1, args.fix_frame): ## only used for save inter_feature 
        for fr in range(0, x.shape[2]-args.fix_frame+1): ## only in 3D seg evalutaion 
            x_ = x[:,:,fr:fr+args.fix_frame,:,:]
            
            with torch.no_grad():
                if args.training_version.startswith('v0'):
                    middle1 = {"image_target": image_target[:,:,fr:fr+args.fix_frame,:,:], "random_selected_class": random_selected_class}
                    if args.text_encoding != "None":
                        middle1["text_features"] = aligned_text_features
                    x_restored, cls_tokens, middle_output = model.forward_encoder(x_, mask_ratio=args.mask_ratio, middle=middle1)
                    pred1 = model.forward_decoder(x_restored, cls_tokens, middle_output)
                    if args.arch_version.startswith('v1'):
                        pred1 = model.unpatchify3D(pred1)
                    preds[:,:,fr:fr+args.fix_frame,:,:]+=pred1
                    cnts[:,:,fr:fr+args.fix_frame,:,:]+=1
    
                    # if args.select_cls == []:
                    #     inter_feature[n_inter:n_inter+1,:,:] = middle_output['x_masked_b'] #x_restored
                    # print("middle_output['x_masked_b']", middle_output['x_masked_b'].shape)
                    n_inter+=1
    elif args.dataset_type == '2D': ## only for 2D model in 3D medical images (3D slice dir)
        for fr in range(0, x.shape[2]): ## only in 3D seg evalutaion 
            x_ = x[:,:,fr,:,:]
            
            with torch.no_grad():
                if args.training_version.startswith('v0'):
                    middle1 = {"image_target": image_target[:,:,fr,:,:], "random_selected_class": random_selected_class}
                    if args.text_encoding != "None":
                        middle1["text_features"] = aligned_text_features
                    x_restored, cls_tokens, middle_output = model.forward_encoder(x_, mask_ratio=args.mask_ratio, middle=middle1)
                    pred1 = model.forward_decoder(x_restored, cls_tokens, middle_output)
                    if args.arch_version.startswith('v1'):
                        pred1 = model.unpatchify(pred1)
                    preds[:,:,fr,:,:]+=pred1
                    cnts[:,:,fr,:,:]+=1
    
                    # if args.select_cls == []:
                    #     inter_feature[n_inter:n_inter+1,:,:] = middle_output['x_masked_b'] #x_restored
                    # print("middle_output['x_masked_b']", middle_output['x_masked_b'].shape)
                    n_inter+=1

    preds = preds/cnts

    # contain [0] is direct gen/seg
    if 0 in args.select_cls:
        image_target = image_target
        preds = preds
    else:
        if args.select_cls == []:
            image_target = image_target
            preds = preds
        else: ## indirect gen/seg
            image_target = x-image_target
            preds = x-preds
            preds[preds < 0] = 0

    # if args.save_video == 1:
    #     save_tensor_3D(preds, args.output_vis+'/masked_results/'+args.load_csv_type+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.reverse)+'.png')
    # elif args.save_video == 2:
    #     save_tensor_3D(image_target, args.output_vis+'/seg_results/'+args.load_csv_type+'/'+case_id+'_test1_image_target_'+class_+'_'+str(args.thre)+'_'+str(args.reverse)+'.png')
    #     save_tensor_3D(preds, args.output_vis+'/seg_results/'+args.load_csv_type+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.thre)+'_'+str(args.reverse)+'.png')

    # if args.select_cls == []:
    #     inter_feature = inter_feature.detach().cpu().numpy()
    #     if args.save_video == 1:
    #         np.save(args.output_vis+'/inter_features/'+args.load_csv_type+'/'+case_id+'.npy', inter_feature)

    # loss_l2 = (preds - image_target) ** 2
    # loss_l2 = loss_l2.mean()

    # loss_l1 = (preds - image_target)
    # loss_l1 = loss_l1.detach().cpu().numpy()
    # loss_l1 = np.abs(loss_l1)

    # p_loss = 0
    # loss_count = 0
    # for i_sl in range(image_target.shape[2]):
    #     image_target_i = image_target[:,:,i_sl,:,:]
    #     pred_i = preds[:,:,i_sl,:,:]
    #     p_loss = p_loss + torch.mean(perceptual_loss(image_target_i.contiguous(), pred_i.contiguous()))
    #     loss_count+=1
    # loss_lpips = p_loss/loss_count
    # loss = (loss_l2 + loss_lpips)

    # ## segmentation evaluation
    # if args.select_cls == [0]:
    #     seg_label_ = (image_target>args.thre)*1.0
    #     seg_preds_ = (preds>args.thre)*1.0
    # else:
    #     seg_label_ = ((x-image_target)>args.thre)*1.0
    #     seg_preds_ = ((x-preds)>args.thre)*1.0

    # print("x.shape, image_target.shape, preds.shape", x.shape, image_target.shape, preds.shape) # torch.Size([1, 3, 112, 224, 224]) torch.Size([1, 3, 112, 224, 224]) torch.Size([1, 3, 112, 224, 224])
    x = x.squeeze().permute(2, 3, 1, 0).detach().cpu().numpy()
    image_target = image_target.squeeze().permute(2, 3, 1, 0).detach().cpu().numpy()
    preds = preds.squeeze().permute(2, 3, 1, 0).detach().cpu().numpy()
    # print("x.shape, image_target.shape, preds.shape", x.shape, image_target.shape, preds.shape) # (224, 224, 112, 3) (224, 224, 112, 3) (224, 224, 112, 3)

    # if len(args.select_cls) > 1:
    #     ## only for just use token for segmentation (direct way)
    #     seg_label = image_target
    #     seg_preds = preds

    #     seg_label = np.mean(seg_label, axis=3)
    #     seg_preds = np.mean(seg_preds, axis=3)
    #     seg_label = (seg_label>args.thre)*1.0
    #     seg_preds = (seg_preds>args.thre)*1.0

    # else:
    #     ## only for just use x-image_target for segmentation (in-direct way)
    #     if args.select_cls == [0]:
    #         seg_label = image_target
    #         seg_preds = preds
    #     else:
    #         seg_label = x-image_target
    #         seg_preds = x-preds

    #     seg_label = np.mean(seg_label, axis=3)
    #     seg_preds = np.mean(seg_preds, axis=3)
    #     seg_label = (seg_label>args.thre)*1.0 ## predefined 0.25
    #     seg_preds = (seg_preds>args.thre)*1.0 ## predefined 0.25
    #     # print("seg_label.shape, seg_preds.shape", seg_label.shape, seg_preds.shape) # (224, 224, 112) (224, 224, 112)
    #     # print("np.unique(seg_label), np.unique(seg_preds)", np.unique(seg_label), np.unique(seg_preds)) # [0. 1.] [False  True]

    seg_label = np.mean(image_target, axis=3)
    seg_preds = np.mean(preds, axis=3)
    seg_label = (seg_label>args.thre)*1.0 ## predefined 0.25
    seg_preds = (seg_preds>args.thre)*1.0 ## predefined 0.25

    min_size = 100
    labeled_image = measure.label(seg_preds, connectivity=1)
    cleaned_image = remove_small_objects(labeled_image, min_size=min_size)
    binary_cleaned_image = cleaned_image > 0
    struct_element = ball(1)
    seg_preds = binary_opening(binary_cleaned_image, struct_element)

    dice = dice_score(seg_preds, seg_label)
    # print("dice", dice)
    # Example usage with the same volumes from the Dice score example
    # voxel_spacing = (1, 1, 1)  # Assuming isotropic voxels of size 1 unit
    # tolerance = 1  # Tolerance distance
    # nsd_value = nsd(seg_preds, seg_label, voxel_spacing=voxel_spacing, tolerance=tolerance)
    try:
        nsd_value, ASD, hausdorff100, hausdorff95, SOverlap = calculate_nsd(seg_preds, seg_label)
    except:
        nsd_value, ASD, hausdorff100, hausdorff95, SOverlap = 0, 0, 0, 0, 0

    if args.save_video == 2:
        save_tensor_3D_np(seg_label, args.output_vis+'/seg_results/'+args.load_csv_type+'/'+case_id+'_seg_label_'+class_+'_'+str(args.thre)+'_'+str(args.reverse)+'.png')
        save_tensor_3D_np(seg_preds, args.output_vis+'/seg_results/'+args.load_csv_type+'/'+case_id+'_seg_preds_'+class_+'_'+str(args.thre)+'_'+str(args.reverse)+'.png')

    # print("dice", dice)
    # print("nsd_value", nsd_value)

    # return loss.detach().cpu().numpy(), loss_l1.mean(), loss_l2.detach().cpu().numpy(), loss_lpips.detach().cpu().numpy(), dice, nsd_value, ASD, hausdorff100, hausdorff95, SOverlap
    return dice, nsd_value, ASD, hausdorff100, hausdorff95, SOverlap


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.num_classes_with_bg = args.num_classes + 1
    args.organ_token_total = 1*args.token_factor*1 + args.token_factor*args.num_classes ## 20+180 = 200
    args.organ_token_selet = args.token_factor*int(args.num_classes_with_bg*args.mask_ratio) #len(random_selected_class) ## 100

    args.select_cls = [int(i) for i in args.select_cls.split(',') if i != ''] ## masked in image1, i.e., keep in image2
    # print("args.select_cls", args.select_cls)

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
    ## segmentation evaluation
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

    ## load img
    img_list = sorted_nicely([i for i in os.listdir(args.load_data_vis_path) if not i.startswith(".")])
    # print(img_list)

    # loss_list = []
    # loss_l1_list = []
    # loss_l2_list = []
    # loss_lpips_list = []

    dice_list, nsd_list, ASD_list, hausdorff100_list, hausdorff95_list, SOverlap_list = [], [], [], [], [], []

    print("args.save_video", args.save_video)
    print("args.select_cls", args.select_cls)

    for img in img_list: #[0:1]:
        data_path = args.load_data_vis_path+'/'+img
        samp_list = sorted_nicely([i_s for i_s in os.listdir(data_path) if not i_s.startswith(".")])
        images = [cv2.imread(data_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in samp_list]
        image = np.stack(images)
        # image = np.transpose(image, (1, 2, 0))
        image = np.float32(image)
        image = (image-image.min())/(image.max()-image.min()+0.00000001)
        # image = image[40:40+args.fix_frame]
    
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
    
        # loss, loss_l1, loss_l2, loss_lpips, dice, nsd_value, ASD, hausdorff100, hausdorff95, SOverlap = gen_one_image(input_img, model_mae, case_id=img, perceptual_loss=perceptual_loss)
        dice, nsd_value, ASD, hausdorff100, hausdorff95, SOverlap = gen_one_image(input_img, model_mae, case_id=img, perceptual_loss=perceptual_loss)
        # loss_list.append(loss)
        # loss_l1_list.append(loss_l1)
        # loss_l2_list.append(loss_l2)
        # loss_lpips_list.append(loss_lpips)

        dice_list.append(dice)
        nsd_list.append(nsd_value)
        ASD_list.append(ASD)
        hausdorff100_list.append(hausdorff100)
        hausdorff95_list.append(hausdorff95)
        SOverlap_list.append(SOverlap)

    # print("loss_list", np.mean(loss_list))
    # print("loss_l1_list", np.mean(loss_l1_list))
    # print("loss_l2_list", np.mean(loss_l2_list))
    # print("loss_lpips_list", np.mean(loss_lpips_list))
    print("dice_list", np.mean(dice_list))
    print("nsd_list", np.mean(nsd_list))
    print("ASD_list", np.mean(ASD_list))
    print("hausdorff100_list", np.mean(hausdorff100_list))
    print("hausdorff95_list", np.mean(hausdorff95_list))
    print("SOverlap_list", np.mean(SOverlap_list))



















    



