import math
import random
from PIL import Image, ImageFilter
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset




IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_NORMALIZE = torchvision.transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
)

def get_data_aug_summary(transforms):
    data_aug = []
    for r in transforms.transforms:
        data_aug.append((str(r.__class__.__name__), r.__dict__))
    return data_aug


def get_basic_transform(scale, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        torchvision.transforms.ToTensor()
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


def get_single_pil_transform(scale, augmentation, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        augmentation,
        torchvision.transforms.ToTensor()
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


def get_single_tensor_transform(scale, augmentation, normalize=None):
    data_aug = [
        torchvision.transforms.Scale(scale),
        torchvision.transforms.ToTensor(),
        augmentation
    ]
    if normalize is not None:
        data_aug.append(normalize)
    return torchvision.transforms.Compose(data_aug)


class RandomRotate90(object):
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = random_rotate_90(input_, self.p)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class BinaryMask(object):
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_[input_ >= self.thresholds] = 1.0
            input_[input_ < self.thresholds] = 0.0
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class Slice1D(object):
    def __init__(self, dim=0, slice_idx=0):
        self.dim = dim
        self.slice_idx = slice_idx

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = torch.unsqueeze(input_[self.slice_idx,:,:], dim=self.dim)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]


class RandomHueSaturation(object):
    def __init__(self, hue_shift=(-180, 180), sat_shift=(-255, 255),
                    val_shift=(-255, 255), u=0.5):
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift
        self.u = u

    def __call__(self, *inputs):
        outputs = []
        for idx, input_ in enumerate(inputs):
            input_ = random_hue_saturation(input_, self.hue_shift,
                self.sat_shift, self.val_shift, self.u)
            outputs.append(input_)
        return outputs if idx > 1 else outputs[0]



def random_rotate_90(pil_img, p=1.0):
    if random.random() < p:
        angle=random.randint(1,3)*90
        if angle == 90:
            pil_img = pil_img.rotate(90)
        elif angle == 180:
            pil_img = pil_img.rotate(180)
        elif angle == 270:
            pil_img = pil_img.rotate(270)
    return pil_img




blurTransform = torchvision.transforms.Lambda(
    lambda img: img.filter(ImageFilter.GaussianBlur(1.5)))