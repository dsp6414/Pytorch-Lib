import math
import random
from PIL import Image, ImageFilter
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

IMAGENET_MU = [0.485, 0.456, 0.406]
IMAGENET_SIGMA = [0.229, 0.224, 0.225]

CIFAR_MU = [0.4914, 0.4822, 0.4465]
CIFAR_SIGMA = [0.2023, 0.1994, 0.2010]

AIRCRAFT_MU = [0.4812, 0.5122, 0.5356]
AIRCRAFT_SIGMA = [0.2187, 0.2118, 0.2441]

def get_transform(size=224, mu=IMAGENET_MU, sigma=IMAGENET_SIGMA, train=False):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma),
        ])
    return transform

