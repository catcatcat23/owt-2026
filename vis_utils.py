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

# sys.path.append('..')
import OWC2_LIB #models_mae_token2
import re
from einops import rearrange

from sklearn.manifold import TSNE
import nibabel as nib

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def load_mp4_as_numpy_array(file_path):
    # Open the video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {file_path}")

    frames = []
    while True:
        # Read a single frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no more frames are available

        # Append the frame to the list (convert BGR to RGB if needed)
        frames.append(frame)

    # Release the video capture object
    cap.release()

    # Convert the list of frames to a NumPy array
    video_array = np.array(frames)

    return video_array

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

def save_tensor_3D(x, save_name, mask = False, norm=False, thre=0.0, label_list=None, is_list=False, print_slice=False):
    concate = 1
    if is_list:
        x = [i.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy() for i in x]
        fr, h, w, c = x[0].shape
        x_np = np.zeros((fr, h, w*len(x), c))
        for i in range(len(x)):
            x_np[:,:,i*w:(i+1)*w,:] = x[i]
        concate = len(x)
    else:
        x_np = x.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy() ## (fr,w,h,c)

    x_np = x_np*(x_np>=thre)

    if mask == False: ## not mask
        if norm == True:
            x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        x_np = (x_np * 255).astype(np.uint8)
    else: ## mask
        x_np = (x_np * 20).astype(np.uint8) ## only for 9 labels

    if print_slice:
        for i in range(0, x_np.shape[0], 5):
            img = x_np[i,:,:,:]
            if label_list is not None:
                # Add text labels
                for j in range(len(label_list)):
                    pos_x = j * w + 10  # Position for text
                    cv2.putText(img, label_list[j], (pos_x, 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imwrite(save_name+"_"+str(i)+".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    x_np = [x_np[i,:,:,:] for i in range(x_np.shape[0])]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_name+".mp4", fourcc, 20.0, (224*concate, 224))
    for img in x_np:
        if label_list is not None:
            # Add text labels to each frame
            for j in range(len(label_list)):
                pos_x = j * w + 10  # Position for text
                cv2.putText(img, label_list[j], (pos_x, 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        out.write(img)
    out.release()

def random_masking(x, selected_classes, args):
    """
    Mask the indexed part of tokens based on selected classes.
    x: [B, 20*classes, c], sequence
    selected_classes: list of selected class indices
    """
    if len(x.shape) == 3:
        B, L, C = x.shape  # batch, length, channels
        mask = torch.ones([B, L], device=x.device)  # initialize mask with ones
    elif len(x.shape) == 4:
        N, B, L, C = x.shape  # batch, length, channels
        mask = torch.ones([N, B, L], device=x.device)  # initialize mask with ones

    tokens_per_class = args.token_factor

    for cls in selected_classes:
        start_idx = cls * tokens_per_class
        end_idx = (cls + 1) * tokens_per_class
        if len(x.shape) == 3:
            mask[:, start_idx:end_idx] = 0  # mask the selected class tokens
        elif len(x.shape) == 4:
            mask[:, :, start_idx:end_idx] = 0  # mask the selected class tokens

    if len(x.shape) == 3:
        x_masked = x[mask == 1].reshape(B, -1, C)
    elif len(x.shape) == 4:
        x_masked = x[mask == 1].reshape(N, B, -1, C)

    return x_masked, mask

def find_closest_tensors(original_tensor, candidates_list, device, k):
    # Convert list of candidate tensors into a tensor with shape [n, 28, 100, 768]
    candidates = candidates_list
    differences = candidates - original_tensor
    squared_diffs = differences ** 2

    # Sum along all dimensions except the first (batch dimension)
    # sum_squared = squared_diffs.sum(dim=(1, 2, 3))
    sum_squared = squared_diffs.mean(dim=(1, 2, 3))
    # Compute L2 norm (Euclidean distance) for each candidate
    # distances = torch.sqrt(sum_squared)
    distances = sum_squared
    # print("distances", distances)
    
    # Find indices of top k smallest distances
    distance_close, indices_close = torch.topk(distances, k=k, largest=False)
    distance_large, indices_large = torch.topk(distances, k=k, largest=True)
    
    return indices_close, distance_close, indices_large, distance_large

def read_slices(args, case_id):
    data_path = args.load_data_vis_path+'/'+case_id
    # print("data_path", data_path)
    samp_list = sorted_nicely([i_s for i_s in os.listdir(data_path) if not i_s.startswith(".")])
    images = [cv2.imread(data_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in samp_list]
    image = np.stack(images)
    image = np.float32(image)
    image = (image-image.min())/(image.max()-image.min()+0.00000001)
    
    mask_path = args.load_label_vis_path+'/'+case_id
    mask_list = sorted_nicely([i_s for i_s in os.listdir(mask_path) if not i_s.startswith(".")])
    masks = [cv2.imread(mask_path+"/"+i_i, cv2.IMREAD_GRAYSCALE) for i_i in mask_list]
    mask = np.stack(masks)
    label = np.float32(mask)
    
    image = torch.tensor(image)
    label = torch.tensor(label)
    # print("image.shape, label.shape", image.shape, label.shape) # torch.Size([112, 224, 224]) torch.Size([112, 224, 224])
    
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
    
    # print("sample['image'].shape, sample['label'].shape", sample['image'].shape, sample['label'].shape) # torch.Size([1, 3, 112, 224, 224]) torch.Size([1, 3, 112, 224, 224])
    return sample

def save_nii(image_target, save_name, thre = 0.0):
	# Fix orientation for correct display in Slicer
	image_target = image_target*(image_target>=thre)
	image_target = image_target.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy()[:,:,:,0]
	image_target = np.transpose(image_target, (2, 1, 0))  # Reorder axes
	# Flip along all axes to fix orientation
	image_target = np.flip(image_target, axis=0)  # Flip along first axis (axial)
	image_target = np.flip(image_target, axis=1)  # Flip along second axis (sagittal) 
	
	# Create nifti image with correct affine matrix for orientation
	affine = np.array([
	    [1, 0, 0, 0],   
	    [0, 1, 0, 0],   
	    [0, 0, 1, 0],
	    [0, 0, 0, 1]
	])
	nii_img = nib.Nifti1Image(image_target, affine)
	nib.save(nii_img, save_name + '.nii.gz')























































