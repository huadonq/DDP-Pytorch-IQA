import numpy as np
import cv2
import numbers
import math
import random
import torch
from scipy.stats import spearmanr, pearsonr
from collections import OrderedDict

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def crop(img, i, j, h, w):
    """
    Crop the given numpy Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i + h, j:j + w, :]

def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)

def pad(img, padding, fill=0, padding_mode='constant'):
    """Pad the given numpy ndarray on all sides with specified padding mode and fill value.
    Args:
        img (numpy ndarray): image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value on the edge of the image
            - reflect: pads with reflection of image (without repeating the last value on the edge)
                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image (repeating the last value on the edge)
                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Returns:
        Numpy image: padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy ndarray. Got {}'.format(type(img)))
    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    if img.shape[2] == 1:
        return cv2.copyMakeBorder(img,
                                  top=pad_top,
                                  bottom=pad_bottom,
                                  left=pad_left,
                                  right=pad_right,
                                  borderType=_cv2_pad_to_str[padding_mode],
                                  value=fill)[:, :, np.newaxis]
    else:
        return cv2.copyMakeBorder(img,
                                  top=pad_top,
                                  bottom=pad_bottom,
                                  left=pad_left,
                                  right=pad_right,
                                  borderType=_cv2_pad_to_str[padding_mode],
                                  value=fill)


def hflip(img):
    """Horizontally flip the given numpy ndarray.
    Args:
        img (numpy ndarray): image to be flipped.
    Returns:
        numpy ndarray:  Horizontally flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    # img[:,::-1] is much faster, but doesn't work with torch.from_numpy()!
    if img.shape[2] == 1:
        return cv2.flip(img, 1)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given numpy ndarray.
    Args:
        img (numpy ndarray): Image to be flipped.
    Returns:
        numpy ndarray:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    if img.shape[2] == 1:
        return cv2.flip(img, 0)[:, :, np.newaxis]
    else:
        return cv2.flip(img, 0)

class CenterCrop:
    """
    Crops the given numpy ndarray at the center.
    Annotations: 
            size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        path, image, score, scene = sample['img_path'], sample['image'], sample['score'],  sample['scene']
        image = center_crop(image, self.size)

        sample = {
            'img_path': path,
            'image': image,
            'score': score,
            'scene': scene,
        }

        return sample


class RandomCrop:
    """
    Crop the given numpy ndarray at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop. 
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[0:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample): 
        path, image, score, scene = sample['img_path'], sample['image'], sample['score'],  sample['scene']
        
        if self.padding is not None:
            image = pad(image, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.shape[1] < self.size[1]:
            image = pad(image, (self.size[1] - image.shape[1], 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.shape[0] < self.size[0]:
            image = pad(image, (0, self.size[0] - image.shape[0]), self.fill,
                        self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        image = crop(image, i, j, h, w)
        
        sample = {
            'img_path': path,
            'image': image,
            'score': score,
            'scene': scene,
        }

        return sample



class RandomHorizontalFlip:
    """
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        path, image, score, scene = sample['img_path'], sample['image'], sample['score'],  sample['scene']
        if random.random() < self.p:
            image =  hflip(image)

        sample = {
            'img_path': path,
            'image': image,
            'score': score,
            'scene': scene
        }

        return sample
     
class RandomVerticalFlip:
    """
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        path, image, score, scene = sample['img_path'], sample['image'], sample['score'],  sample['scene']
        if random.random() < self.p:
            image =  vflip(image)

        sample = {
            'img_path': path,
            'image': image,
            'score': score,
            'scene': scene
        }


        return sample
 

class Resize:
    def __init__(self, resize):
        if isinstance(resize, numbers.Number):
            self.resize = (int(resize), int(resize))
        else:
            self.resize = resize

        self.resize = math.ceil(resize / 32) * 32
        
    def __call__(self, sample):
        path, image, score, scene = sample['img_path'], sample['image'], sample['score'],  sample['scene']

        # h, w, _ = image.shape
        # factor = self.resize / max(h, w)

        # resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        # print(self.resize)
        image = cv2.resize(image, (self.resize, self.resize))      
        sample = {
            'img_path': path,
            'image': image,
            'score': score,
            'scene': scene
        }

        return sample


class Normalize:
    def __init__(self):
        pass

    def __call__(self, sample):
        path, image, score, scene = sample['img_path'], sample['image'], sample['score'],  sample['scene']

        image = image / 255.
        image = image.astype(np.float32)
        sample = {
            'img_path': path,
            'image': image,
            'score': score,
            'scene': scene
        }

        return sample



class iqaRgressionCollater:
    def __init__(self, resize=224):
        self.resize = resize

    def __call__(self, data):
        all_paths = [s['img_path'] for s in data]
        images = [s['image'] for s in data]
        scores = [s['score'] for s in data]
        all_scenes = [s['scene'] for s in data]

        all_images = np.zeros((len(images), self.resize, self.resize, 3),
                              dtype=np.float32)

        all_scores = np.zeros((len(images), 1), dtype=np.float32)

        for i, image in enumerate(images):
            all_images[i, 0:image.shape[0], 0:image.shape[1], :] = image

        all_images = torch.from_numpy(all_images)
        # B H W 3 ->B 3 H W
        all_images = all_images.permute(0, 3, 1, 2)
        

        for i, score in enumerate(scores):
            all_scores[i, :] = score

        all_scores = torch.from_numpy(all_scores)
        
        return {
            'img_path': all_paths,
            'image': all_images,
            'score': all_scores,
            'scene': all_scenes,
        }

class AverageMeter:
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class SprMetricMeter:
    '''
    This is SROCC-PLCC-RMSE Metric Meter.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_score_array = []
        self.gt_score_array = []

        self.srocc = 0
        self.plcc = 0
        self.rmse = 0

    def update(self, pred_score_array, gt_score_array):
        self.pred_score_array = np.append(self.pred_score_array, pred_score_array.data.cpu().numpy())
        self.gt_score_array = np.append(self.gt_score_array, gt_score_array.data.cpu().numpy())

    def compute(self):
        self.srocc, _ = spearmanr(np.squeeze(self.pred_score_array), np.squeeze(self.gt_score_array))
        self.plcc, _ = pearsonr(np.squeeze(self.pred_score_array), np.squeeze(self.gt_score_array))
        self.rmse = np.sqrt(((self.pred_score_array - self.gt_score_array) ** 2).mean())


class iqaRegressionDataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_score = sample['image'], sample['score']
        except StopIteration:
            self.next_input = None
            self.next_score = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_score = self.next_score
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        scores = self.next_score
        self.preload()

        return inputs, scores


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    only load layer parameters which has same layer name and same layer weight shape

    '''

    if not saved_model_path:
        print('💥💥💥 No pretrained model file! 💥💥💥')
        return
    
    saved_state_dict = torch.load(saved_model_path,
                                    map_location=torch.device('cpu'))

    if not isinstance(saved_state_dict, OrderedDict):
        for key, value in saved_state_dict.items():
            if isinstance(value, OrderedDict):
                saved_state_dict = value
                break
        
        filtered_state_dict = {
            name.split('module.')[-1]: weight
            for name, weight in saved_state_dict.items()
            if name.split('module.')[-1] in model.state_dict() and not any(
                excluded_name in name.split('module.')[-1] for excluded_name in excluded_layer_name)
            and weight.shape == model.state_dict()[name.split('module.')[-1]].shape
        }
    else:
        filtered_state_dict = {
            name: weight
            for name, weight in saved_state_dict.items()
            if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name)
            and weight.shape == model.state_dict()[name].shape
        }

    
    if len(filtered_state_dict) == 0:
        print('💥💥💥  No pretrained parameters to load!  💥💥💥')
    else:
        print('😊😊😊  Weight parameters load successfully!  😊😊😊')
        model.load_state_dict(filtered_state_dict, strict=False)

    return