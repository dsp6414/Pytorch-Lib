# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import _init_paths  #import会执行对应模块的代码

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/mnist/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/mnist/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


data,label  = next(iter(train_loader))
#data.permute([0,2,3,1])
#img = torchvision.utils.make_grid(data).numpy()
#import matplotlib.pyplot as plt
#plt.imshow(img)
#torchvision.utils.save_image(data,'1.jpg')

