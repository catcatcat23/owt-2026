# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from datasets.dataset3D import dataset_reader, RandomGenerator
import random

import timm

# assert timm.__version__ == "0.3.2"  # version check
# assert timm.__version__ == "0.3.2"  # version check # comment for H100
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# import models_mae
# import models_mae_token

from engine_pretrain import train_one_epoch

import math
import sys
from typing import Iterable
import util.lr_sched as lr_sched
from PIL import Image
import cv2

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

def eval_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    # model.train(True)
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    # optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): 
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)): ## num_classes !=1

        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
            # lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.num_classes == 1:
            samples, _ = samples
            samples = samples.to(device, non_blocking=True)
        else:
            # if data_iter_step < 5:
                # print("samples['case_name']", samples['case_name']) 
            image = samples['image'].to(device, non_blocking=True)
            label = samples['label'].to(device, non_blocking=True)
            case_name = samples['case_name']
            # if data_iter_step <= 3:
            #     print(case_name)
            # print("image.shape, label.shape", image.shape, label.shape)  # torch.Size([64, 3, 224, 224]), torch.Size([64, 3, 224, 224]) 
            samples = image

            if args.num_classes > 1:
                image_target = image.clone()
                # class_list = list(range(args.num_classes_with_bg)) ## 0,1,2,3,4,5,6,7,8,9 ## tmp test include 0; or list(range(1, args.num_classes_with_bg))
                if args.training_version.startswith('v0'):
                    class_list = list(range(args.num_classes_with_bg))
                elif args.training_version.startswith('v1'):
                    class_list = list(range(1, args.num_classes_with_bg)) ## shouldnt have 0 in v1 training, since unreasonable in the case generating whole slice when input pure 0 image. Meanwhile, using selected_class/masked token for only generating selected organ has already disentangle tokens for specific organs.
                    # class_list = list(range(args.num_classes_with_bg)) 
                random.shuffle(class_list)
                # random_selected_class = class_list[:int(args.num_classes_with_bg*args.mask_ratio)]
                mask_ratio = random.random() * args.mask_ratio
                random_selected_class = class_list[:int(len(class_list)*mask_ratio)]

                for ms in random_selected_class:
                    image_target[label==ms] = 0
                # print("image.shape, image_target.shape", image.shape, image_target.shape)

        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            if args.arch_version.startswith('v0'):
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            else: ## v1, v2, v3
                if args.training_version.startswith('v0'):
                    middle = {"image_target": image_target, "random_selected_class": random_selected_class}
                    loss, pred, middle_output = model(samples, mask_ratio=mask_ratio, middle=middle)#, mask_ratio=args.mask_ratio)
                elif args.training_version.startswith('v1'):
                    random_selected_class2 = list(range(args.num_classes_with_bg)) ## [0,1,2,3,4,5,6,7,8,9]
                    for i in random_selected_class:
                        random_selected_class2.remove(i)
                    middle = {"image_target": samples-image_target, "random_selected_class": random_selected_class2}
                    loss, pred, middle_output = model(image_target, mask_ratio=mask_ratio, middle=middle)#, mask_ratio=args.mask_ratio)

            print("random_selected_class", random_selected_class)
            print("image_target.shape", image_target.shape)
            print("pred.shape", pred.shape)
            save_tensor_3D(samples, args.output_vis+'/test1_image_'+'case'+'xxx'+'.png')
            save_tensor_3D(label, args.output_vis+'/test1_mask_check_'+'case'+'xxx'+'.png', mask=True)
            save_tensor_3D(image_target, args.output_vis+'/test1_image_target_'+'case'+'xxx'+'.png')
            save_tensor_3D(pred, args.output_vis+'/test1_pred_image_'+'case'+'xxx'+'.png')

        # if data_iter_step == 0:
        #     if args.dataset_type == "2D":
        #         # print("random_selected_class", random_selected_class)
        #         # print("case_name", case_name[10])
        #         # Convert the first prediction to a numpy array and save as PNG
        #         pred_image = pred[-1].detach().cpu().numpy()
        #         pred_image = (pred_image * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         pred_image = np.transpose(pred_image, (1, 2, 0))  # Convert from CHW to HWC format
        #         # Save the image
        #         pred_image_pil = Image.fromarray(pred_image)
        #         pred_image_pil.save(args.output_dir+'vis/0_pred.png')
    
        #         image_target_image = image_target[-1].detach().cpu().numpy()
        #         image_target_image = (image_target_image * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         image_target_image = np.transpose(image_target_image, (1, 2, 0))  # Convert from CHW to HWC format
        #         # Save the image
        #         image_target_image_pil = Image.fromarray(image_target_image)
        #         image_target_image_pil.save(args.output_dir+'vis/0_image_target.png')
    
        #         image_pil = image[-1].detach().cpu().numpy()
        #         image_pil = (image_pil * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         image_pil = np.transpose(image_pil, (1, 2, 0))  # Convert from CHW to HWC format
        #         # Save the image
        #         image_pil_pil = Image.fromarray(image_pil)
        #         image_pil_pil.save(args.output_dir+'vis/0_image.png')
    
        #         mask = label[-1].detach().cpu().numpy()
        #         mask = mask*20/255
        #         mask = (mask * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         mask = np.transpose(mask, (1, 2, 0))  # Convert from CHW to HWC format
        #         # Save the image
        #         mask_pil = Image.fromarray(mask)
        #         mask_pil.save(args.output_dir+'vis/0_mask_check.png')
                
        #         mask = label[-1].detach().cpu().numpy()
        #         mask = mask.astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         mask = np.transpose(mask, (1, 2, 0))  # Convert from CHW to HWC format
        #         # Save the image
        #         mask_pil = Image.fromarray(mask)
        #         mask_pil.save(args.output_dir+'vis/0_mask.png')
        #     elif args.dataset_type == "3D":
        #         pred_images = pred[-1].detach().cpu().numpy()
        #         pred_images = (pred_images * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         for i_im in range(pred_images.shape[1]):
        #             pred_image = np.transpose(pred_images[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
        #             # Save the image
        #             pred_image_pil = Image.fromarray(pred_image)
        #             pred_image_pil.save(args.output_dir+'vis/0_pred_'+str(i_im)+'.png')
    
        #         image_target_images = image_target[-1].detach().cpu().numpy()
        #         image_target_images = (image_target_images * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         for i_im in range(image_target_images.shape[1]):
        #             image_target_image = np.transpose(image_target_images[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
        #             # Save the image
        #             image_target_image_pil = Image.fromarray(image_target_image)
        #             image_target_image_pil.save(args.output_dir+'vis/0_image_target_'+str(i_im)+'.png')
    
        #         image_pils = image[-1].detach().cpu().numpy()
        #         image_pils = (image_pils * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         for i_im in range(image_pils.shape[1]):
        #             image_pil = np.transpose(image_pils[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
        #             # Save the image
        #             image_pil_pil = Image.fromarray(image_pil)
        #             image_pil_pil.save(args.output_dir+'vis/0_image_'+str(i_im)+'.png')
    
        #         masks = label[-1].detach().cpu().numpy()
        #         masks = masks*20/255
        #         masks = (masks * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         for i_im in range(masks.shape[1]):
        #             mask = np.transpose(masks[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
        #             # Save the image
        #             mask_pil = Image.fromarray(mask)
        #             mask_pil.save(args.output_dir+'vis/0_mask_check_'+str(i_im)+'.png')
                
        #         # mask = label[-1.detach().cpu().numpy()
        #         # mask = mask.astype(np.uint8)  # Assuming pred is normalized between 0 and 1
        #         # mask = np.transpose(mask, (1, 2, 0))  # Convert from CHW to HWC format
        #         # # Save the image
        #         # mask_pil = Image.fromarray(mask)
        #         # mask_pil.save(args.output_dir+'vis/0_mask.png')

            # exit()
            
        # loss_value = loss.item()
        # if args.vq_version != None:
        #     loss_value_vq = middle_output["vq_loss"].item()
        #     loss = loss + 1.0*middle_output["vq_loss"]
        # if "LPIPS" in args.loss_version:
        #     p_loss_value = middle_output["p_loss"].item()
        #     loss = loss + 1.0*middle_output["p_loss"]

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     continue
        #     # sys.exit(1)

        # if args.vq_version != None:
        #     if not math.isfinite(loss_value_vq):
        #         print("Loss VQ is {}, stopping training".format(loss_value_vq))
        #         continue
        #         # sys.exit(1)

        # loss /= accum_iter
        # loss_scaler(loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        # if (data_iter_step + 1) % accum_iter == 0:
        #     optimizer.zero_grad()

        torch.cuda.synchronize()
        break

    #     metric_logger.update(loss=loss_value)
    #     if args.vq_version != None:
    #         metric_logger.update(loss_vq=loss_value_vq)
    #     if "LPIPS" in args.loss_version:
    #         metric_logger.update(p_loss=p_loss_value)

    #     lr = optimizer.param_groups[0]["lr"]
    #     metric_logger.update(lr=lr)

    #     loss_value_reduce = misc.all_reduce_mean(loss_value)
    #     if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
    #         """ We use epoch_1000x as the x-axis in tensorboard.
    #         This calibrates different curves when batch size changes.
    #         """
    #         epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
    #         log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
    #         log_writer.add_scalar('lr', lr, epoch_1000x)


    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    ## new
    parser.add_argument('--save_freq', type=int, default=20, help='save_freq')
    parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
    parser.add_argument('--arch_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--training_version', type=str, default='v0', help='v0, v1...')
    parser.add_argument('--token_factor', type=int, default=1, help='how many tokens to generate a class')
    parser.add_argument('--loss_version', type=str, default='L2', help='L1-LPIPS-GAN')
    parser.add_argument('--dataset_type', type=str, default='2D', help='2D, 3D') ## but 3D controlled by training_version -3D
    parser.add_argument('--LA', type=bool, default=False, help='False, True')

    parser.add_argument('--output_vis', type=str, default=None)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    # seed = args.seed #+ misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    if args.num_classes == 1:
        # simple augmentation
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)
    else:
        dataset_train = dataset_reader(base_dir=args.data_path, split="train", num_classes=args.num_classes, 
                                    transform=transforms.Compose([RandomGenerator(output_size=[args.input_size, args.input_size], low_res=[128, 128])]), model_args = args)
        print("The length of train set is: {}".format(len(dataset_train)))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_classes == 1:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        # def worker_init_fn(worker_id):
        #     random.seed(args.seed + worker_id)
        data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True,)
                             #worker_init_fn=worker_init_fn)
    
    # define the model
    if args.num_classes == 1:
        import models_mae
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    else:
        if args.arch_version.startswith('v0'):
            import models_mae_token
            model = models_mae_token.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        # elif args.arch_version.startswith('v1'):
        #     import models_mae_token2
        #     model = models_mae_token2.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, model_args=args)
        else: ## v1, v2, v3...
            import OWC2_LIB ## should also include all experiments of OWC2
            model = OWC2_LIB.__dict__[args.model](img_size=args.input_size, norm_pix_loss=args.norm_pix_loss, model_args=args)

    checkpoint = torch.load(args.resume, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
        epoch=2
        data_loader_train.sampler.set_epoch(epoch)
        _ = eval_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
    )
        # if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
        #     misc.save_model(
        #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #         loss_scaler=loss_scaler, epoch=epoch)

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 'epoch': epoch,}

        # if args.output_dir and misc.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    args.num_classes_with_bg = args.num_classes + 1
    args.organ_token_total = 1*args.token_factor*1 + args.token_factor*args.num_classes ## 20+180 = 200
    # args.organ_token_selet = args.token_factor*int(args.num_classes_with_bg*args.mask_ratio) #len(random_selected_class) ## 100

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

    args.loss_version = args.loss_version.split('-') ## loss_dict

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
        args.model = args.model.split("-LA")[0].split("-")[0]

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
