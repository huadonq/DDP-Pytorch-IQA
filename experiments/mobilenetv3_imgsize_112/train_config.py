import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import data_root_dir, pretrained_weight_map
from iqaRegression.dataset import RegressionDataset
from iqaRegression.common import RandomCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize, iqaRgressionCollater, load_state_dict
from iqaRegression import backbones
import torch
import torchvision.transforms as transforms


class config:


    network = 'MobileNetV3_Small'

    input_image_size = 112
    gap = 24
    assert input_image_size % 2 == 0, 'input_image_size must be Even Number!'
    assert gap % 2 == 0, 'gap must be Even Number!'
    
    loss_name = 'L1loss'

    # load pretrained model or not
    pretrained_available = True
    pre_model_path = os.path.join(
        BASE_DIR,
        'pretrained_models',
        pretrained_weight_map[network]
    )

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
    batch_size = 128
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

 