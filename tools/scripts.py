import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import collections
import json
import time
import numpy as np
import re
import shutil
from tqdm import tqdm

from apex import amp
import torch
import torch.nn.functional as F

from iqaRegression.common import iqaRegressionDataPrefetcher, AverageMeter, SprMetricMeter





def all_reduce_operation_in_group_for_variables(variables, operator, group):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables




def train_distill_regression(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train distill regression model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size

    prefetcher = iqaRegressionDataPrefetcher(train_loader)
    images, labels = prefetcher.next()

    iter_index = 1
    while images is not None:
        images = images.cuda()
        labels = labels.cuda()
        tea_outputs, stu_outputs = model(images)

        # loss = criterion(preds, scores)

        loss = 0
        loss_value = {}
        for loss_name in criterion.keys():
            if loss_name in ['CELoss']:
                if not config.freeze_teacher:
                    temp_loss = criterion[loss_name](tea_outputs, labels)
                    loss_value['tea_' + loss_name] = temp_loss
                    loss += temp_loss
                temp_loss = criterion[loss_name](stu_outputs, labels)
                loss_value['stu_' + loss_name] = temp_loss
                loss += temp_loss
            elif loss_name in ['DKDLoss']:
                temp_loss = criterion[loss_name](stu_outputs, tea_outputs,
                                                 labels)
                loss_value[loss_name] = temp_loss
                loss += temp_loss
            else:
                temp_loss = criterion[loss_name](stu_outputs, tea_outputs)
                loss_value[loss_name] = temp_loss
                loss += temp_loss

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            optimizer.zero_grad()
            continue

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.barrier()
        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)

        losses.update(loss, images.size(0))

        images, labels = prefetcher.next()

        if iter_index % config.print_interval == 0:
            log_info = f'🙏train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, total_loss: {loss:.4f}🙏'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1
        

    scheduler.step()

    return losses.avg


def train_distill_classification(train_loader, model, criterion, optimizer,
                                 scheduler, epoch, logger, config):
    '''
    distill classification model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()
    if config.freeze_teacher:
        model.module.teacher.eval()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size

    prefetcher = ClassificationDataPrefetcher(train_loader)
    images, labels = prefetcher.next()
    iter_index = 1

    while images is not None:
        images, labels = images.cuda(), labels.cuda()

        if torch.any(torch.isinf(images)) or torch.any(torch.isinf(labels)):
            continue

        if torch.any(torch.isnan(images)) or torch.any(torch.isnan(labels)):
            continue

        tea_outputs, stu_outputs = model(images)

        loss = 0
        loss_value = {}
        for loss_name in criterion.keys():
            if loss_name in ['CELoss']:
                if not config.freeze_teacher:
                    temp_loss = criterion[loss_name](tea_outputs, labels)
                    loss_value['tea_' + loss_name] = temp_loss
                    loss += temp_loss
                temp_loss = criterion[loss_name](stu_outputs, labels)
                loss_value['stu_' + loss_name] = temp_loss
                loss += temp_loss
            elif loss_name in ['DKDLoss']:
                temp_loss = criterion[loss_name](stu_outputs, tea_outputs,
                                                 labels)
                loss_value[loss_name] = temp_loss
                loss += temp_loss
            else:
                temp_loss = criterion[loss_name](stu_outputs, tea_outputs)
                loss_value[loss_name] = temp_loss
                loss += temp_loss

        if loss == 0. or torch.any(torch.isinf(loss)) or torch.any(
                torch.isnan(loss)):
            optimizer.zero_grad()
            continue

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.barrier()

        for key, value in loss_value.items():
            [value] = all_reduce_operation_in_group_for_variables(
                variables=[value],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss_value[key] = value / float(config.gpus_num)

        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)

        losses.update(loss, images.size(0))

        images, labels = prefetcher.next()

        log_info = ''
        if iter_index % config.print_interval == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, loss: {loss:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value:.4f} '
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    scheduler.step()

    return losses.avg




def test_distill_regression_for_all_dataset(val_loader_list,
                                        model,
                                        criterion,
                                        config
                                        ):
    local_rank = torch.distributed.get_rank()
    teacher_model = model.module.teacher
    student_model = model.module.student
    tea_result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader) in enumerate(
            zip(config.val_dataset_name_list[0], val_loader_list)):

        sub_daset_result_dict = test_iqa_regression(
            per_sub_dataset_loader, teacher_model, criterion, config)

        tea_result_dict[per_sub_dataset_name] = sub_daset_result_dict

    stu_result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader) in enumerate(
            zip(config.val_dataset_name_list[0], val_loader_list)):

        sub_daset_result_dict = test_iqa_regression(
            per_sub_dataset_loader, student_model, criterion, config)

        stu_result_dict[per_sub_dataset_name] = sub_daset_result_dict


    return tea_result_dict, stu_result_dict


def test_text_regression_for_all_dataset(val_loader_list,
                                        model,
                                        criterion,
                                        config
                                        ):
    local_rank = torch.distributed.get_rank()
    
    result_dict = collections.OrderedDict()
    for index, (per_sub_dataset_name, per_sub_dataset_loader) in enumerate(
            zip(config.val_dataset_name_list[0], val_loader_list)):
        # connect_char = "[+]"
        # per_sub_dataset_name = connect_char.join(per_sub_dataset_name)
        # per_sub_dataset_name = per_sub_dataset_name.replace("/", "[s]")

        sub_daset_result_dict = test_iqa_regression(
            per_sub_dataset_loader, model, criterion, config)

        result_dict[per_sub_dataset_name] = sub_daset_result_dict

    return result_dict


def test_iqa_regression(val_loader, model, criterion, config):
    batch_dataload_time = AverageMeter()
    batch_inference_time = AverageMeter()
    losses = AverageMeter()
    Srocc_Plcc_Rmse = SprMetricMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        model_on_cuda = next(model.parameters()).is_cuda
        for data in tqdm(val_loader):
            img_path, images, scores = data['img_path'], data['image'], data['score']
            if model_on_cuda:
                images = images.cuda()
                scores = scores.cuda()
            torch.cuda.synchronize()
            batch_dataload_time.update(time.time() - end)
            end = time.time()
            preds = model(images)
            torch.cuda.synchronize()
            batch_inference_time.update(time.time() - end)

            loss = criterion(preds, scores)
            # loss = sum(loss_dict.values())

            if config.distributed:
                torch.distributed.barrier()
                [loss] = all_reduce_operation_in_group_for_variables(
                    variables=[loss],
                    operator=torch.distributed.ReduceOp.SUM,
                    group=config.group)
                loss = loss / float(config.gpus_num)

            losses.update(loss, images.size(0))

            Srocc_Plcc_Rmse.update(preds, scores)

            end = time.time()

    Srocc_Plcc_Rmse.compute()
    Srocc = Srocc_Plcc_Rmse.srocc
    Plcc = Srocc_Plcc_Rmse.plcc
    Rmse = Srocc_Plcc_Rmse.rmse


    # per image data load time(ms) and inference time(ms)
    per_image_load_time = batch_dataload_time.avg / (config.batch_size //
                                        config.gpus_num) * 1000
    per_image_inference_time = batch_inference_time.avg / (config.batch_size //
                                        config.gpus_num) * 1000

        
    test_loss = losses.avg

    result_dict = {
        'per_image_load_time': per_image_load_time,
        'per_image_inference_time': per_image_inference_time,
        'test_loss': test_loss,
        'SROCC': Srocc,
        'PLCC': Plcc,
        'RMSE': Rmse,
    }
    return result_dict

def train_iqa_regression(train_loader, model, criterion, optimizer, scheduler,
                         epoch, logger, config):
    '''
    train iqa regression model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = torch.distributed.get_rank()
    iters = len(train_loader.dataset) // config.batch_size

    prefetcher = iqaRegressionDataPrefetcher(train_loader)
    images, scores = prefetcher.next()

    iter_index = 1
    while images is not None:
        images = images.cuda()
        scores = scores.cuda()

        preds = model(images)
        loss = criterion(preds, scores)

        if loss == 0.:
            optimizer.zero_grad()
            continue

        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()


        torch.distributed.barrier()
        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)


        losses.update(loss, images.size(0))

        images, scores = prefetcher.next()

        if iter_index % config.print_interval == 0:
            log_info = f'🙏train: epoch {epoch:0>4d}, iter [{iter_index:0>5d}, {iters:0>5d}], lr: {scheduler.get_lr()[0]:.6f}, total_loss: {loss:.4f}🙏'
            logger.info(log_info) if local_rank == 0 else None

        iter_index += 1

    scheduler.step()

    return losses.avg