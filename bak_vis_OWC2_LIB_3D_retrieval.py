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

    return parser

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

def save_tensor_3D(x, save_name, mask = False, norm=False):
    x_np = x.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy() ## (fr,w,h,c)
    if mask == False:
        if norm == True:
            x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-8)
        x_np = (x_np * 255).astype(np.uint8)
    else:
        x_np = (x_np * 20).astype(np.uint8) ## only for 9 labels
    # for i in range(x_np.shape[0]):
    #     cv2.imwrite(save_name+"_"+str(i)+".png", cv2.cvtColor(x_np[i,:,:,:], cv2.COLOR_RGB2BGR))

    x_np = [x_np[i,:,:,:] for i in range(x_np.shape[0])]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_name+".mp4", fourcc, 20.0, (224, 224))
    for img in x_np:
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
    # candidates = torch.stack(candidates_list).to(device)
    candidates = candidates_list
    
    # print("selected_classes", selected_classes)
    # candidates = candidates[:,:,20:,:]
    # original_tensor = original_tensor[:,20:,:]
    # candidates = candidates[:,:,20:40,:]
    # original_tensor = original_tensor[:,20:40,:]
    # candidates = candidates[:,:,:20,:]
    # original_tensor = original_tensor[:,:20,:]
    # candidates, _ = random_masking(candidates, selected_classes, args)
    # original_tensor, _ = random_masking(original_tensor, selected_classes, args)
    # print("candidates.shape, original_tensor.shape", candidates.shape, original_tensor.shape)

    # Compute the difference between each candidate and the original tensor
    differences = candidates - original_tensor
    
    # Square the differences to eliminate negative values
    squared_diffs = differences ** 2
    # squared_diffs = torch.abs(differences) #** 2
    
    # Sum along all dimensions except the first (batch dimension)
    sum_squared = squared_diffs.sum(dim=(1, 2, 3))
    
    # Compute L2 norm (Euclidean distance) for each candidate
    # distances = torch.sqrt(sum_squared)
    distances = sum_squared
    
    # Find indices of top k smallest distances
    _, indices = torch.topk(distances, k=k, largest=False)
    _, indices_large = torch.topk(distances, k=k, largest=True)
    
    return indices, indices_large

def read_slices(args, case_id):
    data_path = args.load_data_vis_path+'/'+case_id
    print("data_path", data_path)
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

def gen_one_image(input_img, model, case_list, inter_feature_list, case_id=0):

    random_selected_class = args.select_cls
    sample1 = input_img[0]
    x_masked_b_all = input_img[1]
    # x_masked_b_all_2 = input_img[2]

    x = sample1['image'].to(device)
    label = sample1['label'].to(device)
    x_masked_b_all = x_masked_b_all.to(device)
    # x_masked_b_all_2 = x_masked_b_all_2.to(device)

    image_target = filter_class(x, label, random_selected_class)

    class_ = 'cls'
    for icl in random_selected_class:
        class_ += str(icl)

    preds = torch.zeros(x.shape).to(device)
    cnts = torch.zeros(x.shape).to(device)

    #### retrieval
    ## mask inter feature 1
    x_masked_b_all, _ = random_masking(x_masked_b_all, random_selected_class, args)
    inter_feature_list = torch.stack(inter_feature_list).to(device)
    inter_feature_list, _ = random_masking(inter_feature_list, random_selected_class, args)
    print("x_masked_b_all.shape, inter_feature_list.shape", x_masked_b_all.shape, inter_feature_list.shape)

    ## x_masked_b_all_2
    n_inter=0
    x_masked_b_all_2 = torch.zeros(x_masked_b_all.shape).to(device)
    for i_b in range(x_masked_b_all.shape[0]):
        # print("i_b", i_b)
        x_masked_b = x_masked_b_all[i_b,:,:]
        x_masked_b = torch.unsqueeze(x_masked_b, dim=0)
        with torch.no_grad():
            if args.training_version.startswith('v0'): ## currently only arch v11 and training v01
                x_masked_ = model.blocks2[0](x_masked_b)
                for bi, blk in enumerate(model.blocks2):
                    if bi > 0:
                        x_masked_ = blk(x_masked_) ## token2 torch.Size([64, 101, 768])
                x_masked_ = model.norm(x_masked_)
                x_masked_ = x_masked_b + x_masked_
                x_masked = x_masked_
                x_restored = x_masked
                pred1 = model.unpatchify3D(model.forward_decoder(x_restored, None, None))
                # print("pred1.shape", pred1.shape) # torch.Size([1, 3, 4, 224, 224])
                preds[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=pred1
                cnts[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=1
                n_inter+=1
                x_masked_b_all_2[i_b,:,:] = x_restored

    closest_indices, indices_large = find_closest_tensors(x_masked_b_all, inter_feature_list, device, 6)
    closest_case = [case_list[c] for c in closest_indices if case_list[c] != case_id] ## top5
    farrest_case = [case_list[c] for c in indices_large if case_list[c] != case_id][:-1] ## top5
    print("closest_indices", closest_indices)
    print("closest_case", closest_case)
    print("farrest_case", farrest_case)

    closest_indices, indices_large = find_closest_tensors(x_masked_b_all_2, inter_feature_2_list, device, 6)
    closest_case = [case_list[c] for c in closest_indices if case_list[c] != case_id] ## top5
    farrest_case = [case_list[c] for c in indices_large if case_list[c] != case_id][:-1] ## top5
    print("closest_indices_2", closest_indices)
    print("closest_case_2", closest_case)
    print("farrest_case_2", farrest_case)

    exit(0)

    n_inter=0
    # for fr in range(0, x.shape[2]-args.fix_frame+1, args.fix_frame):
        # x_ = x[:,:,fr:fr+args.fix_frame,:,:]
    # print("x_masked_b_all.shape[0]", x_masked_b_all.shape[0])
    for i_b in range(x_masked_b_all.shape[0]):
        # print("i_b", i_b)
        x_masked_b = x_masked_b_all[i_b,:,:]
        x_masked_b = torch.unsqueeze(x_masked_b, dim=0)
        
        with torch.no_grad():
            if args.training_version.startswith('v0'): ## currently only arch v11 and training v01
                # middle1 = {"image_target": image_target[:,:,fr:fr+args.fix_frame,:,:], "random_selected_class": random_selected_class}
                # x_restored, cls_tokens, middle_output = model.forward_encoder(x_, mask_ratio=args.mask_ratio, middle=middle1)
                ## (1, 120, 768)
                x_masked_b, mask = model.random_masking(x_masked_b, random_selected_class)
                ## (1, 120-len(random_selected_class)*20, 768)
                x_masked_ = model.blocks2[0](x_masked_b)
                for bi, blk in enumerate(model.blocks2):
                    if bi > 0:
                        x_masked_ = blk(x_masked_) ## token2 torch.Size([64, 101, 768])
                x_masked_ = model.norm(x_masked_)
                x_masked_ = x_masked_b + x_masked_
                x_masked = x_masked_
                x_restored = x_masked
                ## (1, 120-len(random_selected_class)*20, 768)

                ## before latent diffusion, x_restored = self.token_restore(x_masked, mask) ## in which, using initilized and untrained model.mask_token; or just using zeros
                # print(model.mask_token)
                pred1 = model.unpatchify3D(model.forward_decoder(x_restored, None, None))
                preds[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=pred1
                cnts[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=1

                n_inter+=1

    preds = preds/cnts

    save_tensor_3D(preds, args.generate_in_step4+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.reverse)+'.png')

    if random_selected_class == []:
        ## to check the 3D image generated in step2 and step4 are identical
        img_generate_in_step2 = load_mp4_as_numpy_array(args.generate_in_step2+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.reverse)+'.png.mp4')
        img_generate_in_step4 = load_mp4_as_numpy_array(args.generate_in_step2+'/'+case_id+'_test1_pred_image_'+class_+'_'+str(args.reverse)+'.png.mp4')
        # print("Are the arrays identical?", np.array_equal(img_generate_in_step4, img_generate_in_step2))
        assert np.array_equal(img_generate_in_step4, img_generate_in_step2)


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

    # os.makedirs(args.output_vis+'/masked_results/'+args.load_csv_type, exist_ok=True)
    # os.makedirs(args.output_vis+'/inter_features/'+args.load_csv_type, exist_ok=True)
    os.makedirs(args.output_vis+'/generate_in_step4/'+args.load_csv_type, exist_ok=True)

    args.inter_feature_path = args.output_vis+'/inter_features/'+args.load_csv_type+'/'
    args.generate_in_step2 = args.output_vis+'/masked_results/'+args.load_csv_type+'/'
    args.generate_in_step4 = args.output_vis+'/generate_in_step4/'+args.load_csv_type+'/'

    chkpt_dir = args.checkpoint
    model = prepare_model(chkpt_dir, args.model, args, img_size=args.input_size)

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    
    print('Model loaded.')
    
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)

    ## load img
    img_list = sorted_nicely([i for i in os.listdir(args.load_data_vis_path) if (not i.startswith(".")) and (not len(os.listdir(args.load_data_vis_path+"/"+i))==0)])
    # print(img_list)

    ## feature list for retrieval
    random_selected_class = args.select_cls
    case_list = []
    inter_feature_list = []
    inter_feature_2_list = []
    for img in img_list:
        case_list.append(img)

        x_masked_b_all = np.load(args.inter_feature_path + '/' + img + '.npy')
        x_masked_b_all = torch.tensor(x_masked_b_all)

        ## inter_feature_2_list
        x_masked_b_all, _ = random_masking(x_masked_b_all, random_selected_class, args)
        x_masked_b_all = x_masked_b_all.to(device)
        inter_feature_list.append(x_masked_b_all)

        # for ni in range(inter_feature_list.shape[0]):
        # preds = torch.zeros(x.shape).to(device)
        # cnts = torch.zeros(x.shape).to(device)
        n_inter=0
        x_masked_b_all_2 = torch.zeros(x_masked_b_all.shape).to(device)
        for i_b in range(x_masked_b_all.shape[0]):
            # print("i_b", i_b)
            x_masked_b = x_masked_b_all[i_b,:,:]
            x_masked_b = torch.unsqueeze(x_masked_b, dim=0)
            with torch.no_grad():
                if args.training_version.startswith('v0'): ## currently only arch v11 and training v01
                    x_masked_ = model.blocks2[0](x_masked_b)
                    for bi, blk in enumerate(model.blocks2):
                        if bi > 0:
                            x_masked_ = blk(x_masked_) ## token2 torch.Size([64, 101, 768])
                    x_masked_ = model.norm(x_masked_)
                    x_masked_ = x_masked_b + x_masked_
                    x_masked = x_masked_
                    x_restored = x_masked
                    pred1 = model.unpatchify3D(model.forward_decoder(x_restored, None, None))
                    # print("pred1.shape", pred1.shape) # torch.Size([1, 3, 4, 224, 224])
                    # preds[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=pred1
                    # cnts[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=1
                    n_inter+=1
                    x_masked_b_all_2[i_b,:,:] = x_restored

        # preds = preds/cnts
        inter_feature_2_list.append(x_masked_b_all_2)

    inter_feature_list = torch.stack(inter_feature_list).to(device)
    inter_feature_2_list = torch.stack(inter_feature_2_list).to(device)
    print("inter_feature_list.shape, inter_feature_2_list.shape", inter_feature_list.shape, inter_feature_2_list.shape)

    for i in range(len(case_list))[52:]:
        case_id = case_list[i]
        print("----------------------case_id----------------------", case_id)

        x_masked_b_all = inter_feature_list[i]
        x_masked_b_all_2 = inter_feature_2_list[i]

        closest_indices, indices_large = find_closest_tensors(x_masked_b_all, inter_feature_list, device, 6)
        closest_case = [case_list[c] for c in closest_indices if case_list[c] != case_id] ## top5
        farrest_case = [case_list[c] for c in indices_large if case_list[c] != case_id][:-1] ## top5
        print("closest_indices", closest_indices)
        print("closest_case", closest_case)
        print("indices_large", indices_large)
        print("farrest_case", farrest_case)
    
        closest_indices_2, indices_large_2 = find_closest_tensors(x_masked_b_all_2, inter_feature_2_list, device, 6)
        closest_case_2 = [case_list[c] for c in closest_indices_2 if case_list[c] != case_id] ## top5
        farrest_case_2 = [case_list[c] for c in indices_large_2 if case_list[c] != case_id][:-1] ## top5
        print("closest_indices_2", closest_indices_2)
        print("closest_case_2", closest_case_2)
        print("indices_large_2", indices_large_2)
        print("farrest_case_2", farrest_case_2)

        ## use x_masked_b_all_2 as the retrieval results
        closest_indices_2 = closest_indices_2 ## top5 with self
        indices_large_2 = indices_large_2[:-1] ## top5
        for c in closest_indices_2[:3]: ## top3
            case_id_c = case_list[c]
            sample = read_slices(args, case_id_c)
            ## sample['image'].shape, sample['label'].shape torch.Size([1, 3, 112, 224, 224]) torch.Size([1, 3, 112, 224, 224])

            x = sample['image'].to(device)
            label = sample['label'].to(device)
            image_target = filter_class(x, label, random_selected_class)
            class_ = 'cls'
            for icl in random_selected_class:
                class_ += str(icl)

            preds = torch.zeros(x.shape).to(device)
            cnts = torch.zeros(x.shape).to(device)
            for i_b in range(x_masked_b_all.shape[0]):
                x_masked_b = x_masked_b_all[i_b,:,:]
                x_masked_b = torch.unsqueeze(x_masked_b, dim=0)
                with torch.no_grad():
                    if args.training_version.startswith('v0'): ## currently only arch v11 and training v01
                        x_masked_ = model.blocks2[0](x_masked_b)
                        for bi, blk in enumerate(model.blocks2):
                            if bi > 0:
                                x_masked_ = blk(x_masked_) ## token2 torch.Size([64, 101, 768])
                        x_masked_ = model.norm(x_masked_)
                        x_masked_ = x_masked_b + x_masked_
                        x_masked = x_masked_
                        x_restored = x_masked
                        pred1 = model.unpatchify3D(model.forward_decoder(x_restored, None, None))
                        # print("pred1.shape", pred1.shape) # torch.Size([1, 3, 4, 224, 224])
                        preds[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=pred1
                        cnts[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=1
            preds = preds/cnts

            save_tensor_3D(preds, args.generate_in_step4+'/'+case_id+'_test1_pred_image_'+class_+'.png')

            preds = preds.squeeze().permute(1, 2, 3, 0).detach().cpu().numpy()[:,:,:,1]
            break

        break


    exit(0)

    for img in img_list:
        data_path = args.load_data_vis_path+'/'+img
        print("data_path", data_path)
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
        print("image.shape, label.shape", image.shape, label.shape)
    
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
        
        print("sample['image'].shape, sample['label'].shape", sample['image'].shape, sample['label'].shape)

        input_img=[]
        input_img.append(sample)

        x_masked_b_all = np.load(args.inter_feature_path + '/' + img + '.npy')
        x_masked_b_all = torch.tensor(x_masked_b_all)
        input_img.append(x_masked_b_all)

        # x_masked_b_all_2 = np.load(args.inter_feature_path + '/' + img + '_2.npy')
        # x_masked_b_all_2 = torch.tensor(x_masked_b_all_2)
        # input_img.append(x_masked_b_all_2)
        # print("x_masked_b_all.shape, x_masked_b_all_2.shape", x_masked_b_all.shape, x_masked_b_all_2.shape)
        ## 要修改 不能用直接存储的 x_masked_b_all_2, 再 random masking，应该用 x_masked_b_all random masking 再经过block，再对比
    
        gen_one_image(input_img, model, case_list, inter_feature_list, case_id=img)

        break
    



