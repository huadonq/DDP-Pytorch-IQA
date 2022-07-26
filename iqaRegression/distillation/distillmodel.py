import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
import argparse
from iqaRegression import backbones
from iqaRegression.common import load_state_dict
from tools.utils import create_model
from iqaRegression.regression import *

__all__ = [
    'KDModel',
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Text Detection Model Training')
    parser.add_argument(
        '--work-dir',
        default='/home/jovyan/myiqa-master_model/experiments/distillmodel_mobilenetv3_small',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()

class KDModel(nn.Module):

    def __init__(self,
                 teacher_model,
                 student_model,
                 freeze_teacher=True,
                 num_classes=1):
        super(KDModel, self).__init__()
        self.freeze_teacher = freeze_teacher

        self.teacher = teacher_model
        self.student = student_model

        if self.freeze_teacher:
            for m in self.teacher.parameters():
                m.requires_grad = False

    def forward(self, x):
        if self.freeze_teacher:
            with torch.no_grad():
                tea_out = self.teacher(x)
        else:
            tea_out = self.teacher(x)
        stu_out = self.student(x)
        

        return tea_out, stu_out

if __name__ == '__main__':

    

    args = parse_args()
    sys.path.append(args.work_dir)
    print(args.work_dir)
    from train_config import config     
    
    teacher_model = backbones.__dict__[config.teacher_network](**{
        'num_classes': 1,
    })
    inpu_test = torch.randn(config.batch_size, 3, config.input_image_size, config.input_image_size)
    in_features = teacher_model(inpu_test).shape[-1]
    fc = Regression(in_features)
    teacher_model = nn.Sequential(
        teacher_model,
        fc,
    )

    if hasattr(config, 'teacher_model_path'):
        load_state_dict(config.teacher_model_path, teacher_model)


    student_model = backbones.__dict__[config.student_network](**{
        'num_classes': 1,
    })
    inpu_test = torch.randn(config.batch_size, 3, config.input_image_size, config.input_image_size)
    in_features = student_model(inpu_test).shape[-1]
    fc = Regression(in_features)
    student_model = nn.Sequential(
        student_model,
        fc,
    )

    if hasattr(config, 'student_model_path'):
            load_state_dict(config.student_model_path, student_model)


    net = KDModel_my(
                teacher_model,
                student_model,
                freeze_teacher=True,
                num_classes=1)

    image_h, image_w = 112, 112
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, out1_shape: {out[0].shape}, out2_shape: {out[1].shape}')
