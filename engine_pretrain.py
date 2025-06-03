# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import random

import util.misc as misc
import util.lr_sched as lr_sched

import os
from PIL import Image
import numpy as np


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if args.text_encoding != "None":
        text_features = torch.load(args.text_encoding, map_location="cpu").to(device)
        aligned_text_features = text_features.repeat_interleave(args.token_factor, dim=0)  # Shape (100, 512)
        print("aligned_text_features.shape", aligned_text_features.shape)
        # aligned_text_features = aligned_text_features.unsqueeze(0).expand(args.batch_size, -1, -1, -1)  # Shape (B, 100, 512)
        aligned_text_features = aligned_text_features.unsqueeze(0).expand(args.batch_size, -1, -1, -1)  # Shape (B, 100, 512, 768)
        print("aligned_text_features.shape 2", aligned_text_features.shape)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): 
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)): ## num_classes !=1

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

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

        with torch.amp.autocast('cuda'): ## only for H100
            # with torch.cuda.amp.autocast(): ## only for SLURM
            if args.arch_version.startswith('v0'):
                loss, pred, middle_output = model(samples, mask_ratio=args.mask_ratio)
            else: ## v1, v2, v3
                if args.training_version.startswith('v0'):
                    middle = {"image_target": image_target, "random_selected_class": random_selected_class}
                    if args.text_encoding != "None":
                        middle["text_features"] = aligned_text_features
                    loss, pred, middle_output = model(samples, mask_ratio=mask_ratio, middle=middle)#, mask_ratio=args.mask_ratio)
                elif args.training_version.startswith('v1'):
                    random_selected_class2 = list(range(args.num_classes_with_bg)) ## [0,1,2,3,4,5,6,7,8,9]
                    for i in random_selected_class:
                        random_selected_class2.remove(i)
                    middle = {"image_target": samples-image_target, "random_selected_class": random_selected_class2}
                    loss, pred, middle_output = model(image_target, mask_ratio=mask_ratio, middle=middle)#, mask_ratio=args.mask_ratio)

        if data_iter_step == 0:
            if args.dataset_type == "2D":
                # print("random_selected_class", random_selected_class)
                # print("case_name", case_name[10])
                # Convert the first prediction to a numpy array and save as PNG
                pred_image = pred[-1].detach().cpu().numpy()
                pred_image = (pred_image * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                pred_image = np.transpose(pred_image, (1, 2, 0))  # Convert from CHW to HWC format
                # Save the image
                pred_image_pil = Image.fromarray(pred_image)
                pred_image_pil.save(args.output_dir+'vis/0_pred.png')
    
                image_target_image = image_target[-1].detach().cpu().numpy()
                image_target_image = (image_target_image * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                image_target_image = np.transpose(image_target_image, (1, 2, 0))  # Convert from CHW to HWC format
                # Save the image
                image_target_image_pil = Image.fromarray(image_target_image)
                image_target_image_pil.save(args.output_dir+'vis/0_image_target.png')
    
                image_pil = image[-1].detach().cpu().numpy()
                image_pil = (image_pil * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                image_pil = np.transpose(image_pil, (1, 2, 0))  # Convert from CHW to HWC format
                # Save the image
                image_pil_pil = Image.fromarray(image_pil)
                image_pil_pil.save(args.output_dir+'vis/0_image.png')
    
                mask = label[-1].detach().cpu().numpy()
                mask = mask*20/255
                mask = (mask * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                mask = np.transpose(mask, (1, 2, 0))  # Convert from CHW to HWC format
                # Save the image
                mask_pil = Image.fromarray(mask)
                mask_pil.save(args.output_dir+'vis/0_mask_check.png')
                
                mask = label[-1].detach().cpu().numpy()
                mask = mask.astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                mask = np.transpose(mask, (1, 2, 0))  # Convert from CHW to HWC format
                # Save the image
                mask_pil = Image.fromarray(mask)
                mask_pil.save(args.output_dir+'vis/0_mask.png')
            elif args.dataset_type == "3D":
                # if args.arch_version.startswith("v1") or args.arch_version.startswith("v2"):
                #     pred_images = pred[-1].detach().cpu().numpy()
                #     pred_images = (pred_images * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                #     # if (args.dataset_type == "3D" and args.temp_stride == args.fix_frame):
                #     # print("pred_images.shape", pred_images.shape)
                #     pred_image = np.transpose(pred_images, (1, 2, 0))  # Convert from CHW to HWC format
                #     # Save the image
                #     pred_image_pil = Image.fromarray(pred_image)
                #     pred_image_pil.save(args.output_dir+'vis/0_pred_last.png')
                #     # else:
                #     #     for i_im in range(pred_images.shape[1]):
                #     #         pred_image = np.transpose(pred_images[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
                #     #         # Save the image
                #     #         pred_image_pil = Image.fromarray(pred_image)
                #     #         pred_image_pil.save(args.output_dir+'vis/0_pred_'+str(i_im)+'.png')
                # else:
                pred_images = pred[-1].detach().cpu().numpy()
                pred_images = (pred_images * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                for i_im in range(pred_images.shape[1]):
                    pred_image = np.transpose(pred_images[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
                    # Save the image
                    pred_image_pil = Image.fromarray(pred_image)
                    pred_image_pil.save(args.output_dir+'vis/0_pred_'+str(i_im)+'.png')
    
                image_target_images = image_target[-1].detach().cpu().numpy()
                image_target_images = (image_target_images * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                for i_im in range(image_target_images.shape[1]):
                    image_target_image = np.transpose(image_target_images[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
                    # Save the image
                    image_target_image_pil = Image.fromarray(image_target_image)
                    image_target_image_pil.save(args.output_dir+'vis/0_image_target_'+str(i_im)+'.png')
    
                image_pils = image[-1].detach().cpu().numpy()
                image_pils = (image_pils * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                for i_im in range(image_pils.shape[1]):
                    image_pil = np.transpose(image_pils[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
                    # Save the image
                    image_pil_pil = Image.fromarray(image_pil)
                    image_pil_pil.save(args.output_dir+'vis/0_image_'+str(i_im)+'.png')
    
                masks = label[-1].detach().cpu().numpy()
                masks = masks*20/255
                masks = (masks * 255).astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                for i_im in range(masks.shape[1]):
                    mask = np.transpose(masks[:,i_im,:,:], (1, 2, 0))  # Convert from CHW to HWC format
                    # Save the image
                    mask_pil = Image.fromarray(mask)
                    mask_pil.save(args.output_dir+'vis/0_mask_check_'+str(i_im)+'.png')
                
                # mask = label[-1.detach().cpu().numpy()
                # mask = mask.astype(np.uint8)  # Assuming pred is normalized between 0 and 1
                # mask = np.transpose(mask, (1, 2, 0))  # Convert from CHW to HWC format
                # # Save the image
                # mask_pil = Image.fromarray(mask)
                # mask_pil.save(args.output_dir+'vis/0_mask.png')

            # exit()
            
        loss_value = loss.item()
        if args.vq_version != None:
            loss_value_vq = middle_output["vq_loss"].item()
            loss = loss + 1.0*middle_output["vq_loss"]
        if "LPIPS" in args.loss_version:
            p_loss_value = middle_output["p_loss"].item()
            loss = loss + 1.0*middle_output["p_loss"]
        if args.arch_version == 'v02':
            d_loss_value = middle_output["d_loss"].item()
            loss = loss + 1.0*middle_output["d_loss"]
            g_loss_value = middle_output["g_loss"].item()
            loss = loss + 1.0*middle_output["g_loss"]

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            continue
            # sys.exit(1)

        if args.vq_version != None:
            if not math.isfinite(loss_value_vq):
                print("Loss VQ is {}, stopping training".format(loss_value_vq))
                continue
                # sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if args.vq_version != None:
            metric_logger.update(loss_vq=loss_value_vq)
        if "LPIPS" in args.loss_version:
            metric_logger.update(p_loss=p_loss_value)
        if args.arch_version == 'v02':
            metric_logger.update(d_loss=d_loss_value)
            metric_logger.update(g_loss=g_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}