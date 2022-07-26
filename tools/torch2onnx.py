import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import numpy as np

import torch
import onnx
import onnxsim
import onnxruntime
from tools.utils import create_model
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Text Detection Model Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')

    return parser.parse_args()

# onnx version==1.10.2,onnx-simplifier version==0.3.6
def convert_pytorch_model_to_onnx_model(model,
                                        inputs,
                                        save_file_path,
                                        opset_version=13,
                                        use_onnxsim=True):
    print(f'starting export with onnx version {onnx.__version__}...')
    torch.onnx.export(model,
                      inputs,
                      save_file_path,
                      export_params=True,
                      verbose=False,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=['inputs'],
                      output_names=['outputs'],
                      dynamic_axes={
                          'inputs': {},
                          'outputs': {}
                      })

    # load and check onnx model
    onnx_model = onnx.load(save_file_path)
    onnx.checker.check_model(onnx_model)
    print(f'onnx model {save_file_path} is checked!')

    # Simplify onnx model
    if use_onnxsim:
        print(f'using onnx-simplifier version {onnxsim.__version__}...')
        onnx_model, check = onnxsim.simplify(
            onnx_model,
            dynamic_input_shape = False,
            input_shapes={'inputs': inputs.shape})
        assert check, 'assert onnxsim model check failed'
        onnx.save(onnx_model, save_file_path)

        print(
            f'😊😊😊 onnxsim model is checked, convert onnxsim model success, saved as {save_file_path} 😊😊😊'
        )


def main():
    args = parse_args()
    sys.path.append(args.work_dir)
    from test_config import config
    onnx_dir = os.path.join(args.work_dir, 'onnx_pth')
    os.makedirs(onnx_dir) if not os.path.exists(onnx_dir) else None

    model = create_model(config)
    model.eval()

    images = torch.randn(1, 3, config.input_image_size, config.input_image_size)

    convert_pytorch_model_to_onnx_model(model,
                                        images,
                                        config.save_onnx_path,
                                        opset_version=11)
    test_onnx_images = np.random.randn(1, 3, config.input_image_size, config.input_image_size).astype(np.float32)
    model = onnx.load(config.save_onnx_path)
    onnxruntime_session = onnxruntime.InferenceSession(config.save_onnx_path)
    outputs = onnxruntime_session.run(None, dict(inputs=test_onnx_images))

    print('1111,onnx result:', outputs)
if __name__ == "__main__":
    main()