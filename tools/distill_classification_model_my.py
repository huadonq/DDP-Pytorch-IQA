import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format
from iqaRegression import losses
from tools.scripts import test_iqa_regression, train_iqa_regression, test_text_regression_for_all_dataset, test_distill_regression_for_all_dataset, train_distill_regression
from tools.utils import (get_logger, set_seed, worker_seed_init_fn, build_optimizer,
                         build_scheduler, build_training_mode, create_model, compute_macs_and_params)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Text Detection Model Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    set_seed(config.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    # start init process
    
    torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
    torch.cuda.set_device(local_rank)
    config.group = torch.distributed.new_group(list(range(
        config.gpus_num)))

    #mkdir teacher model
    os.makedirs(os.path.join(checkpoint_dir, 'teacher')) if not os.path.exists(os.path.join(checkpoint_dir, 'teacher')) and local_rank == 0 else None
    os.makedirs(os.path.join(checkpoint_dir, 'student')) if not os.path.exists(os.path.join(checkpoint_dir, 'student')) and local_rank == 0 else None

    os.makedirs(checkpoint_dir) if not os.path.exists(checkpoint_dir) and local_rank == 0 else None
    os.makedirs(log_dir) if not os.path.exists(log_dir) and local_rank == 0  else None

  
    torch.distributed.barrier()

    global logger
    logger = get_logger('train', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers

    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    if config.tensorboard_available:
        writer = SummaryWriter(config.tensorboard_log_dir)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=config.seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        config.train_dataset, shuffle=True) 
    train_loader = DataLoader(config.train_dataset,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              pin_memory=False,
                              num_workers=num_workers,
                              collate_fn=config.train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)


    val_loader_list = []
    for per_sub_dataset in config.val_dataset_list:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            per_sub_dataset, shuffle=False) 
        per_sub_loader = DataLoader(per_sub_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=False,
                                    num_workers=num_workers,
                                    collate_fn=config.test_collater,
                                    sampler=val_sampler)
        val_loader_list.append(per_sub_loader)


    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in ['model']:
                log_info = f'{key}: {value}'
                logger.info(log_info)  if local_rank == 0 else None
    # model = create_model(config)

    model = config.net
    
    # macs, params = compute_macs_and_params(config, model)
    # model_profile_info = f'🚀🚀🚀🚀🚀🚀{config.network} -----> FLOPS : {macs}, Params : {params}🚀🚀🚀🚀🚀🚀'
    # logger.info(model_profile_info) if local_rank == 0 else None

    if config.tensorboard_available  and local_rank == 0 :
        input_test = torch.randn(config.batch_size, 3, config.input_image_size, config.input_image_size)
        writer.add_graph(model,input_test)
    
    model = model.cuda()
    train_criterion = config.train_criterion
    test_criterion = config.test_criterion

    for name in train_criterion.keys():
        train_criterion[name] = train_criterion[name].cuda()
    test_criterion = test_criterion.cuda()

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    for name, param in model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    for name, buffer in model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    model = build_training_mode(config, model, optimizer)

    start_epoch, train_time = 1, 0
    tea_srocc, tea_plcc, tea_rmse, tea_loss = 0.0, 0.0, 0.0, 0.0
    stu_srocc, stu_plcc, stu_rmse, stu_loss = 0.0, 0.0, 0.0, 0.0
    tea_best_srocc, tea_best_plcc, tea_best_rmse = 0.0, 0.0, 0.0
    tea_best_weight_loss = 0.0
    stu_best_srocc, stu_best_plcc, stu_best_rmse = 0.0, 0.0, 0.0
    stu_best_weight_loss = 0.0
    # automatically resume model for training if checkpoint model exist
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        used_time = checkpoint['time']
        train_time += used_time
        
        best_srocc, best_plcc, best_rmse, loss, lr = checkpoint[
            'best_srocc'], checkpoint['best_plcc'], checkpoint[
                'best_rmse'], checkpoint['loss'], checkpoint['lr']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.4f} hours, best_srocc: {best_srocc:.4f}, best_plcc: {best_plcc:.4f}, best_rmse: {best_rmse:.4f}, loss: {loss:.3f}, lr: {lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {scheduler.get_lr()[0]}'
        logger.info(log_info)  if local_rank == 0 else None
        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch) 
        train_loss = train_distill_regression(train_loader, model, train_criterion,
                                          optimizer, scheduler, epoch, logger,
                                          config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

        if config.tensorboard_available  and local_rank == 0 :
            writer.add_scalars('epoch losses/train', {'epoch train loss': train_loss}, epoch + 1)

        torch.cuda.empty_cache()

        # if epoch % config.save_interval == 0:
        #     if local_rank == 0 :
        #         torch.save(model.module.state_dict(),
        #                     os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))

        if epoch % config.test_interval == 0 :
            teacher_result_dict, student_result_dict = test_distill_regression_for_all_dataset(val_loader_list,
                                        model,
                                        test_criterion,
                                        config
                                    )
            torch.cuda.empty_cache()

            log_info = f'🚀🚀🚀🚀🚀🚀🚀🚀🚀   eval_teacher: epoch: {epoch:0>3d}   🚀🚀🚀🚀🚀🚀🚀🚀🚀\n'
            total_result = None
            for key, value in teacher_result_dict.items():
                total_result = value
                break
            if total_result is not None:
                for key, value in total_result.items():
                    if key in [
                            'per_image_load_time', 'per_image_inference_time'
                    ]:
                        log_info += f'{key}: {value:.3f}ms\n'
                    elif key in ['test_loss']:
                        log_info += f'{key}: {value:.3f}\n'
                    else:
                        log_info += f'{key}: {value:.3f}\n'
                tea_srocc, tea_plcc, tea_rmse, tea_loss = total_result['SROCC'], total_result[
                'PLCC'], total_result['RMSE'], total_result['test_loss']

                if config.tensorboard_available and local_rank == 0 :
                    writer.add_scalars('epoch losses/teacher_val', {'epoch val loss': tea_loss}, epoch + 1) 
                    writer.add_scalars('epoch metric/teacher_SROCC', {'epoch val SROCC': tea_srocc}, epoch + 1)
                    writer.add_scalars('epoch metric/teacher_PLCC', {'epoch val PLCC': tea_plcc}, epoch + 1)
                    writer.add_scalars('epoch metric/teacher_RMSE', {'epoch val RMSE': tea_rmse}, epoch + 1)

            logger.info(log_info) if local_rank == 0 else None

            log_info = f'🚀🚀🚀🚀🚀🚀🚀🚀🚀   eval_student: epoch: {epoch:0>3d}   🚀🚀🚀🚀🚀🚀🚀🚀🚀\n'
            total_result = None
            for key, value in student_result_dict.items():
                total_result = value
                break
            if total_result is not None:
                for key, value in total_result.items():
                    if key in [
                            'per_image_load_time', 'per_image_inference_time'
                    ]:
                        log_info += f'{key}: {value:.3f}ms\n'
                    elif key in ['test_loss']:
                        log_info += f'{key}: {value:.3f}\n'
                    else:
                        log_info += f'{key}: {value:.3f}\n'
                stu_srocc, stu_plcc, stu_rmse, stu_loss = total_result['SROCC'], total_result[
                'PLCC'], total_result['RMSE'], total_result['test_loss']

                if config.tensorboard_available and local_rank == 0 :
                    writer.add_scalars('epoch losses/student_val', {'epoch val loss': stu_loss}, epoch + 1) 
                    writer.add_scalars('epoch metric/student_SROCC', {'epoch val SROCC': stu_srocc}, epoch + 1)
                    writer.add_scalars('epoch metric/student_PLCC', {'epoch val PLCC': stu_plcc}, epoch + 1)
                    writer.add_scalars('epoch metric/student_RMSE', {'epoch val RMSE': stu_rmse}, epoch + 1)

            logger.info(log_info) if local_rank == 0 else None

                

        train_time += (time.time() - per_epoch_start_time) / 3600

        if epoch % config.test_interval == 0:
            # save best f1 model and each epoch checkpoint
            weight_loss = 0.6 * tea_srocc + 0.4 * tea_plcc
            
            if weight_loss > tea_best_weight_loss:
                log_info = f'😊😊😊    updating ---> the best srocc plcc rmse    😊😊😊'
                logger.info(log_info) if local_rank == 0 else None
                torch.save(model.module.teacher.state_dict(),
                           os.path.join(checkpoint_dir, 'teacher', 'best.pth'))

                tea_best_weight_loss = weight_loss
                tea_best_srocc = tea_srocc
                tea_best_plcc = tea_plcc
                tea_best_rmse = tea_rmse

            log_info = f'tea_best_srocc: {tea_best_srocc:.4f}, tea_best_plcc: {tea_best_plcc:.4f}, tea_best_rmse: {tea_best_rmse:.4f}\n'
            logger.info(log_info) if local_rank == 0 else None


            weight_loss = 0.6 * stu_srocc + 0.4 * stu_plcc
            
            if weight_loss > stu_best_weight_loss:
                log_info = f'😊😊😊    updating ---> the best srocc plcc rmse    😊😊😊'
                logger.info(log_info) if local_rank == 0 else None
                torch.save(model.module.student.state_dict(),
                           os.path.join(checkpoint_dir, 'student', 'best.pth'))

                tea_best_weight_loss = weight_loss
                tea_best_srocc = tea_srocc
                tea_best_plcc = tea_plcc
                tea_best_rmse = tea_rmse

            log_info = f'stu_best_srocc: {tea_best_srocc:.4f}, stu_best_plcc: {tea_best_plcc:.4f}, stu_best_rmse: {tea_best_rmse:.4f}\n'
            logger.info(log_info) if local_rank == 0 else None

            torch.save(
                {
                    'epoch': epoch,
                    'time': train_time,
                    'tea_best_srocc': tea_best_srocc,
                    'tea_best_plcc': tea_best_plcc,
                    'tea_best_rmse': tea_best_rmse,
                    'tea_loss': tea_loss,
                    'stu_best_srocc': stu_best_srocc,
                    'stu_best_plcc': stu_best_plcc,
                    'stu_best_rmse': stu_best_rmse,
                    'stu_loss': stu_loss,
                    'lr': scheduler.get_lr()[0],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth'))

    
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')) and  local_rank == 0 :
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(
                checkpoint_dir,
                f'best_srocc{best_srocc:.3f}_plcc{best_plcc:.3f}_rmse{best_rmse:.3f}.pth'
            ))

    log_info = f'train done. train time: {train_time:.4f} hours, best_srocc: {best_srocc:.4f}, best_plcc: {best_plcc:.4f}, best_rmse: {best_rmse:.4f}'
    logger.info(log_info)  if local_rank == 0 else None

    return

if __name__ == '__main__':
    main()