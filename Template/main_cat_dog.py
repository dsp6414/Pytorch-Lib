
# coding: utf-8
import _init_paths

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

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import utils
from utils.Tensor import to_variable,to_numpy
from utils.metric import AccuracyMeter,MovingAverageMeter,AverageMeter
from torch.autograd import Variable
import torch.nn.functional as F
import image_loader
from image_loader import image_loader

from PIL import Image


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, valid_loader, use_cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.use_cuda = use_cuda

    def train(self, epoch):
        self.model.train()

        train_loss = AverageMeter()
        train_acc = AccuracyMeter()

        for i, (x, y) in enumerate(self.train_loader):
            x = Variable(x)
            y = Variable(y)
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(float(loss.data))

            y_pred = output.data.max(dim=1)[1]
            correct = int(y_pred.eq(y.data).cpu().sum())
            train_acc.update(correct, x.size(0))
            if i % 10 ==0:
                print('\nTrain Epoch/batch| [{}/{}]: Average batch loss: {:.6f}\n'.format(epoch,i,train_acc.accuracy))
        return train_loss.average, train_acc.accuracy

    def validate(self):
        self.model.eval()

        valid_loss = AverageMeter()
        valid_acc = AccuracyMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x = Variable(x, volatile=True)
            y = Variable(y).long()
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            output = self.model(x)
            loss = F.cross_entropy(output, y)

            valid_loss.update(float(loss.data))

            y_pred = output.data.max(dim=1)[1]
            correct = int(y_pred.eq(y.data).cpu().sum())
            valid_acc.update(correct, x.size(0))
        print('\nTrain Epoch [{}]: Average batch loss: {:.6f}\n'.format(epoch,valid_acc.accuracy))
        return valid_loss.average, valid_acc.accuracy

def print_msg(proc,epoch,loss,acc):
    print('proc={},epoch={}:loss={},acc={}'.format(proc,epoch,loss,acc))

if __name__ == '__main__':
    train_batch_size=40

    ROOT_DIR = os.getcwd()
    DATA_HOME_DIR = ROOT_DIR + '/data/cat_dog'

    # paths
    data_path = DATA_HOME_DIR  
    train_path = data_path + '/train/'
    valid_path = data_path + '/test/'
    train_loader,valid_loader=image_loader(train_path,valid_path,
                   train_batch_size,
                   valid_batch_size=None,
                   train_shuffle=True,
                   valid_shuffle=False,
                   train_num_workers=0,
                   valid_num_workers=0)

    #model = models.resnet34(pretrained=True)   
    model = models.resnet50(pretrained=True)  
    for param in model.parameters():
            param.requires_grad = False
    #model.fc = nn.Linear(512,2)
    model.fc = nn.Linear(2048,2)
    #optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4, weight_decay=1e-4)
    if torch.cuda.is_available():
        model.cuda()
    
     #optimizer = optim.Adam(model.module.fc.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    trainer = Trainer(model=model,optimizer=optimizer,train_loader=train_loader,valid_loader=valid_loader,use_cuda=True)
    epochs = 30
    train_loss_epochs=np.zeros((epochs,))
    train_acc_epochs=np.zeros((epochs,))
    test_loss_epochs=np.zeros((epochs,))
    test_acc_epochs=np.zeros((epochs,))
    i=0
    for epoch in range(1, epochs+1):
        scheduler.step()
        train_loss, train_acc=trainer.train(epoch)
        train_loss_epochs[i]=train_loss
        train_acc_epochs[i]=train_acc
        print_msg('train',epoch,train_loss,train_acc)
        test_loss, test_acc=trainer.validate()
        test_loss_epochs[i]=test_loss
        test_acc_epochs[i]=test_acc
        print_msg('test',epoch,test_loss,test_acc)
        i+=1
    data  = dict()
    data['train_loss']=train_loss_epochs
    data['train_acc']=train_acc_epochs
    data['test_loss']=test_loss_epochs
    data['test_acc']=test_acc_epochs
    torch.save(data,'data.pt')
    print('finished')
  