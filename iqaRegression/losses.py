import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn


__all__ = [
    'IQALoss',
    'L1Loss',
    'L2Loss',
]

class IQALoss(nn.Module):
    def __init__(self, loss_name):
      super(IQALoss, self).__init__()
      self.loss_name = loss_name

    def forward(self, preds, targets):
        if self.loss_name == 'L1loss':
            return torch.abs(preds - targets).mean()
        elif self.loss_name == 'L2loss':
            return torch.pow(preds - targets, 2).mean()
        # elif loss_name == 'SmoothL1loss':

# def smooth_l1_loss(input, target, sigma, reduce=True, normalizer=1.0):
#     beta = 1. / (sigma ** 2)
#     diff = torch.abs(input - target)
#     cond = diff < beta
#     loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
#     if reduce:
#         return torch.sum(loss) / normalizer
#     return torch.sum(loss, dim=1) / normalizer
    

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, preds, targets):

        return torch.abs(preds - targets).mean()

    
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, preds, targets):

        return torch.pow(preds - targets, 2).mean()



if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

    import random
    import numpy as np
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from tools.path import iqa_image_dir, iqa_regression_txt_file_path
    from iqaRegression.dataset import RegressionDataset
    from iqaRegression.common import RandomCrop, RandomHorizontalFlip, Normalize, Resize, CenterCrop, iqaRgressionCollater
    import torchvision.transforms as transforms
    from tqdm import tqdm

    train_transform = transforms.Compose([
        Resize(256), 
        RandomCrop(224), 
        RandomHorizontalFlip(), 
        Normalize(),
    ])

    iqaRegressionDataset = RegressionDataset(iqa_regression_txt_file_path, iqa_image_dir, train_transform)


    from torch.utils.data import DataLoader
    train_collate = iqaRgressionCollater(224)
    
    train_loader = DataLoader(iqaRegressionDataset,
                              batch_size=16,
                              shuffle=True,
                             num_workers=16,
                              collate_fn=iqaRgressionCollater())

    from iqaRegression.models.repvgg import create_RepVGG_A0
    net = create_RepVGG_A0(deploy=False)
    loss = L1Loss()
    loss_1 = nn.L1Loss()

    count = 0
    for data in tqdm(train_loader):
        paths, images, score = data['img_path'], data['image'], data['score']
        out = net(images)
        loss_dict = loss(out, score)
        loss_dict_1 = loss_1(out, score)
        print("1111", loss_dict, loss_dict_1)

