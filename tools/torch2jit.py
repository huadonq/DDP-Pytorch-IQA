import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

from tools.utils import create_model
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Text Detection Model Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()

def main():
    args = parse_args()
    sys.path.append(args.work_dir)
    from test_config import config
    jit_dir = os.path.join(args.work_dir, 'jit_pth')
    os.makedirs(jit_dir) if not os.path.exists(jit_dir) else None

    model = create_model(config)
    model.eval()
    images = torch.randn(1, 3, config.input_image_size, config.input_image_size)
    pt_model = torch.jit.trace(model.cpu().eval(), (images))
    torch.jit.save(pt_model, config.save_jit_path)
    
    print('😊😊😊  torch to jit successfully!  😊😊😊')

if __name__ == "__main__":
    main()