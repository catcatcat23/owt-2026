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
            # print("image.shape, label.shape", image.shape, label.shape)  # torch.Size([64, 3, 224, 224]), torch.Size([64, 3, 224, 224]) 
            samples = image

            if args.num_classes > 1:
                image_target = image.clone()
                class_list = list(range(args.num_classes_with_bg)) ## 0,1,2,3,4,5,6,7,8,9 ## tmp test include 0; or list(range(1, args.num_classes_with_bg))
                random.shuffle(class_list)
                random_selected_class = class_list[:int(args.num_classes_with_bg*args.mask_ratio)]
                # print("random_selected_class", random_selected_class)

                for ms in random_selected_class:
                    image_target[label==ms] = 0
                # print("image.shape, image_target.shape", image.shape, image_target.shape)

        with torch.cuda.amp.autocast():
            if args.arch_version == 'v0':
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            elif args.arch_version == 'v1':
                middle = {"image_target": image_target, "random_selected_class": random_selected_class}
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio, middle=middle)#, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

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