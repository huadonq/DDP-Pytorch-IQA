import os
import cv2
import copy
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from iqaRegression.common import RandomCrop, RandomHorizontalFlip, Normalize, Resize, CenterCrop
from iqaRegression.common import iqaRgressionCollater



class TextDetection(Dataset):
    def __init__(self,
                 root_dir,
                 set_name=[
                     'art',
                     'das',
                     'densecurved',
                     'hw_book',
                     'hw_other',
                     'hw_paper',
                     'idensetext',
                     'mlt',
                     'mtwi',
                     'verticaltext',
                     'caigou_20211111_Chinese_01_PPT翻拍',
                     'caigou_20211111_Chinese_02_文档翻拍',
                     'caigou_20211111_Chinese_03_自然光拍照',
                     'caigou_20211111_English_01_PPT翻拍',
                     'caigou_20211111_English_02_文档翻拍',
                     'caigou_20211111_English_03_自然光拍照',
                 ],
                 set_type='train',
                 transform=None):
        self.transform = transform

        assert set_type in ['train', 'test'], 'Wrong set type!'

        image_dir_list = []
        for per_set_name in set_name:
            per_set_image_dir = os.path.join(
                os.path.join(root_dir, per_set_name), set_type)
            image_dir_list.append(per_set_image_dir)

        label_path_list = []
        for per_set_name in set_name:
            per_set_path = os.path.join(root_dir, per_set_name)
            if set_type == 'train':
                per_set_label_path = os.path.join(per_set_path,
                                                  per_set_name + '_train.json')
            elif set_type == 'test':
                per_set_label_path = os.path.join(per_set_path,
                                                  per_set_name + '_test.json')
            label_path_list.append(per_set_label_path)

        image_path_list, image_shape_dict, image_scene_dict = [], {}, {}
        for per_set_image_dir_path, per_set_label_path in tqdm(
                zip(image_dir_list, label_path_list)):
            with open(per_set_label_path, 'r', encoding='UTF-8') as json_f:
                per_set_label = json.load(json_f)
                for key, value in tqdm(per_set_label.items()):
                    per_image_path = os.path.join(per_set_image_dir_path, key)
                    if not os.path.exists(per_image_path):
                        continue

                    shape, scene = value['shape'], value['scene']
                    new_shape = copy.deepcopy(shape)

                    for i in range(len(new_shape)):
                        new_shape[i]['box'] = np.array(
                            new_shape[i]['box']).astype(np.float32)

                    image_path_list.append(per_image_path)
                    image_shape_dict[key] = new_shape
                    image_scene_dict[key] = scene

        self.image_path_list, self.image_shape_dict, self.image_scene_dict = image_path_list, image_shape_dict, image_scene_dict

        print(f"Dataset Num:{len(self.image_path_list)}")

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        path = self.image_path_list[idx]

        image = self.load_image(idx)
        shape = self.load_shape(idx)
        scene = self.load_scene(idx)

        size = [image.shape[0], image.shape[1]]

        sample = {
            'path': path,
            'image': image,
            'shape': shape,
            'scene': scene,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        '''
        convert RGB image to gray image
        '''
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        return image

    def load_shape(self, idx):
        image_name = self.image_path_list[idx].split('/')[-1]
        shape = self.image_shape_dict[image_name]

        return shape

    def load_scene(self, idx):
        image_name = self.image_path_list[idx].split('/')[-1]
        scene = self.image_scene_dict[image_name]

        return scene


class RegressionDataset(Dataset):
    """
    RegressionDataset
    
    Args:
        txt_file: a 2-column txt_file, column one contains the names of image files, column 2 contains the MOS or DMOS Score.
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, 
                root_dir,
                set_name=['koniq10k'],
                set_type='train',
                transform=None
    ):

        assert set_type in ['train', 'test'], 'Wrong Set Type!'

        self.root_dir = root_dir
        self.transform = transform

        
        self.image_path_list = []
        self.score_list = []
        self.image_scene_dict = {}
        for per_set_name in set_name:
            per_set_txt_file = os.path.join(
                os.path.join(root_dir, per_set_name, 'labels'), 
                set_type + '.txt'
            )
            f = open(per_set_txt_file, encoding = "utf-8").readlines()
            for line in f:
                '''
                processing img_path and score
                '''
                img_path = os.path.join(os.path.join(self.root_dir, per_set_name, 'images'), line.split('\t')[1])
                self.image_path_list.append(img_path)
                self.score_list.append(np.array(float(line.split('\t')[-1].split('\n')[0])).astype('float').reshape(-1))
                self.image_scene_dict[img_path] = per_set_name


       

    def __len__(self):

        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_path = self.load_path(idx)
        image = self.load_image(idx)
        score = self.load_score(idx)
        scene = self.load_scene(idx)

        sample = {
            'img_path': img_path, 
            'image': image, 
            'score': score,
            'scene': scene,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


    def load_score(self, idx):
        score = self.score_list[idx]
        
        return score

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        return image
    
    def load_path(self, idx):
        path = self.image_path_list[idx]
        
        return path

    def load_scene(self, idx):
        image_name = self.image_path_list[idx]
        scene = self.image_scene_dict[image_name]

        return scene

class RegressionDataset_no_list(Dataset):
    """
    RegressionDataset
    
    Args:
        txt_file: a 2-column txt_file, column one contains the names of image files, column 2 contains the MOS or DMOS Score.
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, txt_file, root_dir, transform=None):
        f = open(txt_file,encoding = "utf-8")
        txt_read = f.readlines()
        self.root_dir = root_dir
        self.transform = transform

        self.image_path_list = []
        self.score_list = []

        for line in txt_read:
            '''
            processing img_path and score
            '''

            self.image_path_list.append(os.path.join(self.root_dir,line.split('\t')[1]))
            self.score_list.append(np.array(float(line.split('\t')[-1].split('\n')[0])).astype('float').reshape(-1))

    def __len__(self):

        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_path = self.load_path(idx)
        image = self.load_image(idx)
        score = self.load_score(idx)
    
        sample = {
            'img_path': img_path, 
            'image': image, 
            'score': score,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


    def load_score(self, idx):
        score = self.score_list[idx]
        
        return score

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        return image
    
    def load_path(self, idx):
        path = self.image_path_list[idx]
        
        return path




if __name__ == '__main__':
    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    print(BASE_DIR)

    # from tools.path import text_detection_datasets_path

    import random
    import torch
    import numpy as np
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # from simpleTextDetection.common import MainDirectionRandomRotate, RandomRotate, Resize, Normalize, RandomShrink, TextDetectionCollater, DBNetTextDetectionCollater
    import torchvision.transforms as transforms
    from tqdm import tqdm

    # from tools.path import iqa_image_dir, iqa_regression_txt_file_path

    train_transform = transforms.Compose([
        Resize(256), 
        RandomCrop(224), 
        RandomHorizontalFlip(), 
        Normalize(),
    ])
    path_j = '/home/jovyan/myiqa-master/data/all_set/'
    iqaRegressionDataset = RegressionDataset(
                                                path_j, 
                                                set_name=['koniq10k'],
                                                set_type='train',
                                                transform = train_transform
                                            )
    
    
    count = 0
#     for per_sample in tqdm(iqaRegressionDataset):
# # 'img_id': img_name, 'image': image_pair, 'annotations': score_pair
#         print(per_sample['img_id'])
#         print(per_sample['image'].shape)
#         print(per_sample['annotations'])
#         break

    from torch.utils.data import DataLoader
    from tools.utils import worker_seed_init_fn
    
    train_collate = iqaRgressionCollater(224)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     iqaRegressionDataset, shuffle=True)
    train_sampler = None
    train_loader = DataLoader(iqaRegressionDataset, batch_size=128, shuffle=(train_sampler is None),
            num_workers=8,
            collate_fn=train_collate,
            sampler=train_sampler
            )
    # train_loader = DataLoader(iqaRegressionDataset, batch_size=128, shuffle=True, num_workers=8)

    for data in tqdm(train_loader):
        
        img_path, img, score = data['img_path'], data['image'], data['score']
        print(img.shape,score.shape)

        # print(per_sample['image'].shape, per_sample['image'].dtype)
        # print('1111', per_sample['shape'][0])

        # if not os.path.exists('./temp1'):
        #     os.makedirs('./temp1')

        # shape = per_sample['shape']
        # for per_shape in shape:
        #     box = per_shape['box']
        #     box = np.array(box, np.int32)
        #     box = box.reshape((-1, 1, 2))
        #     cv2.polylines(per_sample['image'],
        #                   pts=[box],
        #                   isClosed=True,
        #                   color=(0, 255, 0),
        #                   thickness=3)

        # cv2.imencode(
        #     '.jpg',
        #     per_sample['image'])[1].tofile(f'./temp1/sample_{count}.jpg')

        # if count < 5:
        #     count += 1
        # else:
        #     break

    # from torch.utils.data import DataLoader
    # collater = TextDetectionCollater(resize=960)
    # train_loader = DataLoader(textdetectiondataset,
    #                           batch_size=1,
    #                           shuffle=False,
    #                           num_workers=1,
    #                           collate_fn=collater)

    # count = 0
    # for data in tqdm(train_loader):
    #     images, shapes = data['image'], data['shape']
    #     print(images.shape, images.dtype)
    #     print('2222', shapes[0][0])

    #     if count < 5:
    #         count += 1
    #     else:
    #         break

    # textdetectiondataset = TextDetection(
    #     text_detection_datasets_path,
    #     set_name=[
    #         # 'art',
    #         # 'das',
    #         # 'densecurved',
    #         # 'hw_book',
    #         # 'hw_other',
    #         # 'hw_paper',
    #         # 'idensetext',
    #         # 'mlt',
    #         # 'mtwi',
    #         # 'verticaltext',
    #         # 'caigou_20211111_Chinese_01_PPT翻拍',
    #         # 'caigou_20211111_Chinese_02_文档翻拍',
    #         # 'caigou_20211111_Chinese_03_自然光拍照',
    #         # 'caigou_20211111_English_01_PPT翻拍',
    #         # 'caigou_20211111_English_02_文档翻拍',
    #         # 'caigou_20211111_English_03_自然光拍照',
    #         'eval4_dataset',
    #     ],
    #     set_type='test',
    #     transform=transforms.Compose([
    #         # RandomRotate(angle=[-45, 45], prob=1.0),
    #         Resize(resize=960),
    #         Normalize(),
    #     ]))

    # # count = 0
    # # for per_sample in tqdm(textdetectiondataset):
    # #     print(per_sample['image'].shape, per_sample['image'].dtype)
    # #     print('1111', per_sample['shape'][0])

    # #     if not os.path.exists('./temp1'):
    # #         os.makedirs('./temp1')

    # #     per_sample['image'] = per_sample['image'] * 255
    # #     shape = per_sample['shape']
    # #     for per_shape in shape:
    # #         box = per_shape['box']
    # #         box = np.array(box, np.int32)
    # #         box = box.reshape((-1, 1, 2))
    # #         cv2.polylines(per_sample['image'],
    # #                       pts=[box],
    # #                       isClosed=True,
    # #                       color=(0, 255, 0),
    # #                       thickness=3)

    # #     cv2.imencode(
    # #         '.jpg',
    # #         per_sample['image'])[1].tofile(f'./temp1/sample_{count}.jpg')

    # #     if count < 0:
    # #         count += 1
    # #     else:
    # #         break

    # from torch.utils.data import DataLoader
    # collater = DBNetTextDetectionCollater(resize=960,
    #                                       min_box_size=3,
    #                                       min_max_threshold=[0.3, 0.7],
    #                                       shrink_ratio=0.6)
    # train_loader = DataLoader(textdetectiondataset,
    #                           batch_size=1,
    #                           shuffle=False,
    #                           num_workers=1,
    #                           collate_fn=collater)

    # count = 0
    # for data in tqdm(train_loader):
    #     paths, images, shapes = data['path'], data['image'], data['shape']
    #     # print(images.shape, images.dtype)
    #     # print('2222', shapes['shape'][0][0])

    #     if not os.path.exists('./temp2'):
    #         os.makedirs('./temp2')

    #     for i in range(images.shape[0]):
    #         path = paths[i]
    #         image_name = path.split("/")[-1].split(".")[0]
    #         image = images[i].permute(1, 2, 0).numpy() * 255

    #         shape = shapes['shape'][i]
    #         for per_shape in shape:
    #             box = per_shape['box']
    #             box = np.array(box, np.int32)
    #             box = box.reshape((-1, 1, 2))
    #             cv2.polylines(image,
    #                           pts=[box],
    #                           isClosed=True,
    #                           color=(0, 255, 0),
    #                           thickness=3)

    #         probability_mask = shapes['probability_mask'][i].numpy() * 255
    #         probability_ignore_mask = shapes['probability_ignore_mask'][
    #             i].numpy() * 255
    #         threshold_mask = shapes['threshold_mask'][i].numpy() * 255
    #         threshold_ignore_mask = shapes['threshold_ignore_mask'][i].numpy(
    #         ) * 255

    #         # print("1111", image_name, image.shape, probability_mask.shape,
    #         #       probability_ignore_mask.shape, threshold_mask.shape,
    #         #       threshold_ignore_mask.shape)
    #         # print("2222", probability_mask[0][0:5] / 255.,
    #         #       probability_ignore_mask[0][0:5] / 255.,
    #         #       threshold_mask[0][0:5] / 255.,
    #         #       threshold_ignore_mask[0][0:5] / 255.)

    #         cv2.imencode('.jpg', image)[1].tofile(f'./temp2/{image_name}.jpg')
    #         cv2.imencode('.jpg', probability_mask)[1].tofile(
    #             f'./temp2/{image_name}_probability_mask.jpg')
    #         cv2.imencode('.jpg', probability_ignore_mask)[1].tofile(
    #             f'./temp2/{image_name}_probability_ignore_mask.jpg')
    #         cv2.imencode('.jpg', threshold_mask)[1].tofile(
    #             f'./temp2/{image_name}_threshold_mask.jpg')
    #         cv2.imencode('.jpg', threshold_ignore_mask)[1].tofile(
    #             f'./temp2/{image_name}_threshold_ignore_mask.jpg')

    #     if count < 10:
    #         count += 1
    #     else:
    #         break