import sys
import os
import os.path
import random
import collections
import shutil
import time
import glob
import csv
import numpy as np

from torch.utils import data
from torchvision import datasets, transforms






def image_loader(train_path,valid_path,
                   train_batch_size,
                   valid_batch_size=None,
                   train_shuffle=True,
                   valid_shuffle=False,
                   train_num_workers=0,
                   valid_num_workers=0):

    if valid_batch_size is None:
        valid_batch_size = train_batch_size

    transform =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225] )
        ])
   

    valid_transform =transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    train_loader = data.DataLoader(datasets.ImageFolder(train_path,
                                                   transform=transform),
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle,
                                   num_workers=train_num_workers,drop_last=True)

    valid_loader = data.DataLoader(datasets.ImageFolder(valid_path,
                                                    transform=valid_transform),
                                   batch_size=valid_batch_size,
                                   shuffle=valid_shuffle,
                                   num_workers=valid_num_workers)

    return train_loader, valid_loader


