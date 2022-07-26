import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import data_root_dir, pretrained_weight_map
from iqaRegression.dataset import RegressionDataset
from iqaRegression.common import RandomCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip, Normalize, iqaRgressionCollater, load_state_dict



import torch
import torchvision.transforms as transforms


class config:
    # network = 'MobileNetV3_large_x0_50'
    

    network = 'MobileNetV3_Small'

    input_image_size = 112
    gap = 24
    assert input_image_size % 2 == 0, 'input_image_size must be Even Number!'
    assert gap % 2 == 0, 'gap must be Even Number!'

    loss_name = 'L1loss'
    # load pretrained model or not
    pretrained_test_available = True
    pre_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'checkpoints',
        [pth for pth in os.listdir(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),'checkpoints')) if 'best' in pth][0]
    )    
     
    test_transform = transforms.Compose([
        Resize(input_image_size + gap), 
        RandomCrop(input_image_size), 
        RandomHorizontalFlip(), 
        Normalize(),
    ])

    test_collater = iqaRgressionCollater(input_image_size)
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
    distributed = True
    save_bad_case = False
    save_map = False


    #convert
    # torch2onnx
    save_onnx_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'onnx_pth',
        os.path.dirname(os.path.abspath(__file__)).split('/')[-1] +'_onnx.pth'
    ) 

    # torch2jit
    save_jit_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'jit_pth',
        os.path.dirname(os.path.abspath(__file__)).split('/')[-1] +'_jit.pth'
    ) 
