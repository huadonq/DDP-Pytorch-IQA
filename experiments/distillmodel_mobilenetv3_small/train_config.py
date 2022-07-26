import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import data_root_dir, pretrained_weight_map
from iqaRegression.dataset import RegressionDataset
from iqaRegression.common import RandomCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize, iqaRgressionCollater, load_state_dict
from iqaRegression import backbones
from iqaRegression.distillation import losses
from iqaRegression.regression import *
from iqaRegression.distillation.distillmodel import * 
import torch
import torchvision.transforms as transforms



class config:


    teacher_network = 'MobileNetV3_Large'
    student_network = 'MobileNetV3_Small'

    input_image_size = 112
    gap = 24
    assert input_image_size % 2 == 0, 'input_image_size must be Even Number!'
    assert gap % 2 == 0, 'gap must be Even Number!'
    batch_size = 128
    loss_name = 'L1loss'

     # load pretrained model or not
    teacher_model_path = '/home/jovyan/myiqa-master_model/experiments/mobilenetv3_large_imgsize_192/checkpoints/best.pth'


    student_model_path = '/home/jovyan/myiqa-master_model/experiments/mobilenetv3_imgsize_192/checkpoints/best.pth'
    
    freeze_teacher = True
    teacher_model = backbones.__dict__[teacher_network](**{
        'num_classes': 1,
    })
    inpu_test = torch.randn(batch_size, 3, input_image_size, input_image_size)
    in_features = teacher_model(inpu_test).shape[-1]
    fc = Regression(in_features)
    teacher_model = nn.Sequential(
        teacher_model,
        fc,
    )

    load_state_dict(teacher_model_path, teacher_model)


    student_model = backbones.__dict__[student_network](**{
        'num_classes': 1,
    })
    in_features = student_model(inpu_test).shape[-1]
    fc = Regression(in_features)
    student_model = nn.Sequential(
        student_model,
        fc,
    )
   
    load_state_dict(student_model_path, student_model)


    net = KDModel(
                teacher_model,
                student_model,
                freeze_teacher=True,
                num_classes=1)




    loss_list = ['L2Loss']
    alpha = 1.0
    beta = 0.5
    T = 1.0
    train_criterion = {}
    for loss_name in loss_list:
        if loss_name in ['KDLoss', 'DMLLoss']:
            train_criterion[loss_name] = losses.__dict__[loss_name](T)
        elif loss_name in ['DKDLoss']:
            train_criterion[loss_name] = losses.__dict__[loss_name](alpha,
                                                                    beta, T)
        else:
            train_criterion[loss_name] = losses.__dict__[loss_name]()
    test_criterion = losses.__dict__['CELoss']()



   
    #tensorboard_setting
    tensorboard_available = True
    tensorboard_log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tensorboard_log'
    )    

    
    train_transform = transforms.Compose([
        Resize(input_image_size + gap), 
        RandomCrop(input_image_size), 
        RandomHorizontalFlip(), 
        Normalize(),
    ])

    # test_transform = transforms.Compose([
    #     Resize(input_image_size + gap), 
    #     RandomCrop(input_image_size), 
    #     RandomHorizontalFlip(), 
    #     Normalize(),
    # ])
    train_collater = iqaRgressionCollater(input_image_size)
    test_collater = iqaRgressionCollater(input_image_size)


    train_dataset = RegressionDataset(data_root_dir, transform =  train_transform)
      
    # 完整数据集必须在list中第0个位置
    val_dataset_name_list = [
        [
            'koniq10k',
        ],
    ]
    val_dataset_list = []
    for per_sub_val_dataset_name in val_dataset_name_list:
        per_sub_val_dataset = RegressionDataset(
            data_root_dir,
            set_name = per_sub_val_dataset_name,
            set_type = 'test',
            transform = transforms.Compose([
                Resize(input_image_size + gap), 
                RandomCrop(input_image_size), 
                Normalize(),
            ]))
        
        val_dataset_list.append(per_sub_val_dataset)


    seed = 0
    # batch_size is total size
    
    # num_workers is total workers
    num_workers = 16

    # choose 'SGD' or 'AdamW'
    # optimizer = (
    #     'SGD',
    #     {
    #         'lr': 0.1,
    #         'momentum': 0.9,
    #         'weight_decay': 1e-4,
    #     },
    # )

    optimizer = (
        'AdamW',
        {
            'lr': 1e-3,
            'weight_decay': 0,
        },
    )

    scheduler = (
        'MultiStepLR',
        {
            'warm_up_epochs': 20,
            'gamma': 0.1,
            'milestones': [40],
        },
    )

    # scheduler = (
    #     'CosineLR',
    #     {
    #         'warm_up_epochs': 0,
    #     },
    # )

    epochs = 110
    save_interval = 20
    test_interval = 1
    print_interval = 20

    # only in DistributedDataParallel mode can use sync_bn
    distributed = True
    sync_bn = False
    apex = True

 