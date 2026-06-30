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

from sklearn.manifold import TSNE
import nibabel as nib

from vis_utils import *

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
    parser.add_argument('--tnse_plot', type=int, default=0, help='0=False, 1=True')
    parser.add_argument('--topk', type=int, default=3, help='3,5,7...')
    parser.add_argument('--id_index', type=int, default=0, help='id_index')

    parser.add_argument('--text_encoding', type=str, default="None", help='None or path of text_encoding')

    return parser

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

    os.makedirs(args.output_vis+'/generate_in_step4/'+args.load_csv_type, exist_ok=True)

    args.inter_feature_path = args.output_vis+'/inter_features/'+args.load_csv_type+'/'
    args.generate_in_step2 = args.output_vis+'/masked_results/'+args.load_csv_type+'/'
    args.generate_in_step4 = args.output_vis+'/generate_in_step4/'+args.load_csv_type+'/'

    chkpt_dir = args.checkpoint
    model = prepare_model(chkpt_dir, args.model, args, img_size=args.input_size)

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim = -1)
    
    print('Model loaded.')
    
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)

    ## load img
    img_list = sorted_nicely([i for i in os.listdir(args.load_data_vis_path) if (not i.startswith(".")) and (not len(os.listdir(args.load_data_vis_path+"/"+i))==0)])

    ## feature list for retrieval
    random_selected_class = args.select_cls
    class_ = 'cls'
    for icl in random_selected_class:
        class_ += str(icl)
    case_list = []
    inter_feature_list = []
    for img in img_list:
        case_list.append(img)

        x_masked_b_all = np.load(args.inter_feature_path + '/' + img + '.npy')
        x_masked_b_all = torch.tensor(x_masked_b_all)

        ## inter_feature_list integration
        x_masked_b_all, _ = random_masking(x_masked_b_all, random_selected_class, args)
        x_masked_b_all = x_masked_b_all.to(device)
        inter_feature_list.append(x_masked_b_all)

    inter_feature_list = torch.stack(inter_feature_list).to(device)
    print("inter_feature_list.shape", inter_feature_list.shape)

    if args.tnse_plot == 1:
        #### save tsne plot
        inter_feature_list_tsne = inter_feature_list.mean(dim=1)
        print("inter_feature_list_tsne.shape", inter_feature_list_tsne.shape)
        inter_feature_list_tsne = inter_feature_list_tsne.view(inter_feature_list_tsne.shape[0], -1)
        print("inter_feature_list_tsne.shape", inter_feature_list_tsne.shape)

        # Define color palette
        color_palette = ['#595959', '#C04E15', '#51938B', '#2E85B7', '#FCD46E']

        #### put all token groups in one space
        inter_feature_list_sub = []
        if inter_feature_list.shape[2] > args.token_factor:
            for tf in range(int(inter_feature_list.shape[2]/args.token_factor)):
                inter_feature_list_sub.append(inter_feature_list[:,:,tf*args.token_factor:(tf+1)*args.token_factor,:].mean(dim=1).view(inter_feature_list.shape[0], -1))
            inter_feature_list_sub = torch.stack(inter_feature_list_sub).to(device)
            print("inter_feature_list_sub.shape", inter_feature_list_sub.shape)
            
            # Create discrete color labels based on first dimension before reshaping
            colors = np.repeat(np.arange(inter_feature_list_sub.shape[0]), inter_feature_list_sub.shape[1])
            
            # Reshape to (shape[0]*shape[1], shape[2])
            inter_feature_list_sub = inter_feature_list_sub.view(-1, inter_feature_list_sub.shape[-1])
            print("inter_feature_list_sub reshaped shape", inter_feature_list_sub.shape)

            # Generate and save TSNE plots with discrete colors
            # 2D plot
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=2000, random_state=5, verbose=1)
            tsne_results_2d = tsne.fit_transform(inter_feature_list_sub.detach().cpu().numpy())
            
            plt.figure(figsize=(10, 10))
            unique_colors = np.unique(colors)
            for i, color_val in enumerate(unique_colors):
                mask = colors == color_val
                plt.scatter(tsne_results_2d[mask, 0], tsne_results_2d[mask, 1], 
                        # label=f'Group {i}', alpha=0.6)
                        color=color_palette[i], label=f'Group {i}', alpha=0.6)
            
            plt.savefig(args.generate_in_step4+"/tsne_token_groups_2d_"+str(args.fix_frame)+"_"+class_+".png", dpi=300, bbox_inches='tight')
            plt.close()

            # 3D plot
            tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=2000, random_state=5, verbose=1)
            tsne_results_3d = tsne.fit_transform(inter_feature_list_sub.detach().cpu().numpy())
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, color_val in enumerate(unique_colors):
                mask = colors == color_val
                ax.scatter(tsne_results_3d[mask, 0], tsne_results_3d[mask, 1], tsne_results_3d[mask, 2],
                        # label=f'Group {i}', alpha=0.6)
                        color=color_palette[i], label=f'Group {i}', alpha=0.6)
            
            plt.savefig(args.generate_in_step4+"/tsne_token_groups_3d_"+str(args.fix_frame)+"_"+class_+".png", dpi=300, bbox_inches='tight')
            plt.close()

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=2000, random_state=5, verbose=1)
        tsne_results = tsne.fit_transform(inter_feature_list_tsne.detach().cpu().numpy())  # Shape: [700, 2]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
        for i in range(0, len(case_list), 10):  # Annotate every 10th point
            plt.text(tsne_results[i, 0], tsne_results[i, 1], str(i)+", "+case_list[i], fontsize=4, alpha=0.7)

        plt.savefig(args.generate_in_step4+"/tsne_visualization_"+str(args.fix_frame)+"_"+class_+".png", dpi=300, bbox_inches='tight')

        ## Clean TSNE
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
        plt.savefig(args.generate_in_step4+"/tsne_visualization_"+str(args.fix_frame)+"_"+class_+"_clean.png", dpi=300, bbox_inches='tight')

    for i in range(len(case_list))[args.id_index:]:
        case_id = case_list[i]
        os.makedirs(args.generate_in_step4+'/'+str(args.id_index)+'/', exist_ok = True)
        print("----------------------case_id----------------------", case_id)

        x_masked_b_all = inter_feature_list[i]

        indices_close, distance_close, indices_large, distance_large = find_closest_tensors(x_masked_b_all, inter_feature_list, device, args.topk+1)
        closest_case = [case_list[c] for c in indices_close if case_list[c] != case_id] ## top5
        farthest_case = [case_list[c] for c in indices_large if case_list[c] != case_id] ## top6
        print("----------------------Retriveal----------------------")
        print("indices_close", indices_close)
        print("distance_close", distance_close)
        print("closest_case", closest_case)
        print("indices_large", indices_large)
        print("distance_large", distance_large)
        print("farthest_case", farthest_case)

        ## gen
        sample = read_slices(args, case_id)
        x = sample['image'].to(device)
        label = sample['label'].to(device)
        image_target = filter_class(x, label, random_selected_class)
        save_nii(image_target, args.generate_in_step4+'/'+str(args.id_index)+'/'+case_id+'_test1_pred_image_top'+str(args.topk)+'_'+class_+'_target.png', thre = 0.0) #0.02

        class_ = 'cls'
        for icl in random_selected_class:
            class_ += str(icl)

        ## use x_masked_b_all (indices_close) as the retrieval results
        pred_list = []
        pred_list.append(x)
        pred_list.append(image_target)

        label_list = ["original", "target"]

        indices_close = indices_close ## self + topk
        indices_large = indices_large ## farthest
        indices_close = torch.cat((indices_close, indices_large[0:1]), dim = 0) ## topk + largest one
        for c in indices_close: ## topk
            case_id_c = case_list[c]
            sample = read_slices(args, case_id_c)

            x = sample['image'].to(device)
            label = sample['label'].to(device)
            image_target = filter_class(x, label, random_selected_class)
            class_ = 'cls'
            for icl in random_selected_class:
                class_ += str(icl)

            preds = torch.zeros(x.shape).to(device)
            cnts = torch.zeros(x.shape).to(device)
            for i_b in range(inter_feature_list[c].shape[0]):
                x_masked_b = inter_feature_list[c][i_b,:,:]
                x_masked_b = torch.unsqueeze(x_masked_b, dim=0)
                with torch.no_grad():
                    x_masked_ = model.blocks2[0](x_masked_b)
                    for bi, blk in enumerate(model.blocks2):
                        if bi > 0:
                            x_masked_ = blk(x_masked_)
                    x_masked_ = model.norm(x_masked_)
                    x_masked_ = x_masked_b + x_masked_
                    x_masked = x_masked_
                    x_restored = x_masked
                    pred1 = model.forward_decoder(x_restored, None, None)
                    pred1 = model.unpatchify3D(pred1)
                    preds[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=pred1
                    cnts[:,:,i_b*args.fix_frame:(i_b+1)*args.fix_frame,:,:]+=1
            preds = preds/cnts
            pred_list.append(preds)
            c = str(c.detach().cpu().numpy())
            label_list.append(c)
            save_nii(preds, args.generate_in_step4+'/'+str(args.id_index)+'/'+case_id+'_test1_pred_image_top'+str(args.topk)+'_'+class_+'_'+case_id_c+'_'+str(c)+'.png', thre = 0.0) #0.02

        save_tensor_3D(pred_list, args.generate_in_step4+'/'+str(args.id_index)+'/'+case_id+'_test1_pred_image_top'+str(args.topk)+'_'+class_+'_combined.png', \
            thre=0.0, label_list=label_list, \
            is_list=True, print_slice=True) #0.02

        break



