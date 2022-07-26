

# Environments

**This repository only support DDP training.**

**environments:**
ubuntu1~18.04, 4*Tesla V100, Python Version:3.8.10, CUDA Version:11.4

Please make sure your Python version>=3.7.
**Use pip or conda to install those Packages:**
```
torch==1.10.0
torchvision==0.11.1
torchaudio==0.10.0
onnx==1.11.0
onnx-simplifier==0.3.6
numpy
Cython
pycocotools
opencv-python
tqdm
thop
yapf
tensorboard
apex
```

**How to install apex?**

apex needs to be installed separately.For torch1.10,modify apex/apex/amp/utils.py:
```
if cached_x.grad_fn.next_functions[1][0].variable is not x:
```
to
```
if cached_x.grad_fn.next_functions[0][0].variable is not x:
```

Then use the following orders to install apex:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```
Using apex to train can reduce video memory usage by 25%-30%, but the training speed will be slower, the trained model has the same performance as not using apex.

# Prepare datasets

If you want to reproduce my pretrained models, you need download koniq10k dataset or other datasets but make sure the folder architecture as follows:
```
data
|
|----->koniq10k
|----------------->images
|---------------------------xxx.jpg
...
|---------------------------xxx.jpg
|----------------->labels
|---------------------------train.txt
|---------------------------test.txt
...
|----->other dataset
|----------------->images
|---------------------------xxx.jpg
...
|---------------------------xxx.jpg
|----------------->labels
|---------------------------train.txt
|---------------------------test.txt

txt_file: contains 2-column, column one contains the names of image files, column 2 contains the MOS or DMOS Score.
train.txt for training dataset, test.txt for test dataset.

```

# Train and test model

If you want to train or test model,you need enter a training folder directory,then run train.sh and test.sh.

For example,you can enter experiments/mobilenetv3_imgsize_112.
If you want to train this model from scratch, please delete checkpoints and log folders first,then run train.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../tools/train.py  --work-dir ./
```

CUDA_VISIBLE_DEVICES is used to specify the gpu ids for this training.Please make sure the number of nproc_per_node equal to the number of gpu cards.
Make sure master_addr/master_port are unique for each training.

if you want to test this model,you need have a pretrained model first,modify pre_model_path in test_config.py,then run test.sh:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr 127.0.1.0 --master_port 10000 ../../tools/test.py --work-dir ./
```
Also, You can modify super parameters in train_config.py/test_config.py.

if you want to convert this model to jit, you need have a pretrained model first, modify pre_model_path and save_jit_path in test_config.py,then run torch2jit.sh:
```
python ../../tools/torch2jit.py --work-dir ./
```
Also, You can modify super parameters in test_config.py.

if you want to convert this model to onnx, you need have a pretrained model first, modify pre_model_path and save_onnx_path in test_config.py,then run torch2onnx.sh:
```
python ../../tools/torch2onnx.py --work-dir ./
```
Also, You can modify super parameters in test_config.py.

# IQA-Regression training results

## Koniq10k training results

| Network              | macs     | params      | input size | gpu num      | batch     | warm up | lr decay  | apex | syncbn | epochs | SROCC  |  PLCC  |
| -------------        | -------- | ----------- | ---------- | ------------ | --------- | ------- | --------  | ---- | ------ | ------ | ------ | ------ |
| MUSIQ                | 372.36G  | 125.541M    | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 86.88  | 86.58  |
| Vgg16                | 15.347G  | 14.740M     | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 87.53  | 89.33  |
| Vgg16_Nopretrain     | 15.347G  | 14.740M     | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 74.56  | 77.34  |
| Shufflenetv2         | 33.714M  | 143.329K    | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 72.37  | 75.48  |
| Shufflenetv2_Rank    | 33.714M  | 143.329K    | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 56.71  | 60.37  |
| Mobilenet_v3_small   | 60.864M  | 955.233K    | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 67.64  | 72.71  |
| Regnet_y_400mf       | 417.743M | 3.925M      | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 70.28  | 72.56  |
| RepVgg-A0            | 1.529G   | 7.891M      | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 75.46  | 79.88  |
| MobileOne            | 1.119G   | 4.318M      | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 38.84  | 39.17  |
| RepShufflenetV2_0    | 22.109M  | 53.017K     | 224x224    |4 Tesla V100  | 128       | 0       | multistep | True | False  | 100    | 60.31  | 64.16  |

You can find more model training details in experiments.


# Distillation training results

## Koniq10k training results

**KD loss**
Paper:https://arxiv.org/abs/1503.02531

**DKD loss**
Paper:https://arxiv.org/abs/2203.08679

**DML loss**
Paper:https://arxiv.org/abs/1706.00384

to be continue...

You can find more model training details in experiments/distillmodel_mobilenetv3_small/.

# Citation

If you find my work useful in your research, please consider citing:
```
@inproceedings{zjh,
 title={SimpleAICV-IQA},
 author={zjh},
 year={2022}
}
```