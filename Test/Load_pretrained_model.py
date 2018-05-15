# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function


import _init_paths  #import会执行对应模块的代码

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from Tensor import to_variable,to_numpy
from metric import AccuracyMeter,MovingAverageMeter,AverageMeter

import unittest
_dummy = unittest.TestCase('__init__')
assert_equal = _dummy.assertEqual

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #(None,20,20,20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return x


class New_Net(nn.Module):

    def __init__(self):
        super(New_Net, self).__init__()
        self.conv0 = nn.Conv2d(1, 20, kernel_size=5,padding=2)
        self.conv1 = nn.Conv2d(20, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #(None,20,20,20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Sequential(
            nn.Linear(320, 64),
             nn.Linear(64, 10))
        

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.conv0(x))
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return x


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, valid_loader, use_cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.use_cuda = use_cuda

    def train(self, epoch):
        self.model.train()

        train_loss = MovingAverageMeter()
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
            if i % 100 ==0:
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

            valid_loss.update(float(loss.data), x.size(0))

            y_pred = output.data.max(dim=1)[1]
            correct = int(y_pred.eq(y.data).cpu().sum())
            valid_acc.update(correct, x.size(0))
        print('\nTrain Epoch [{}]: Average batch loss: {:.6f}\n'.format(epoch,valid_acc.accuracy))
        return valid_loss.average, valid_acc.accuracy

def load_sub_modules_from_pretrained(pretraind_sub_modules_list,model_sub_modules_list):
    for p,m in zip(pretraind_sub_modules_list,model_sub_modules_list):
        for param_p,param_m in zip(p.parameters(),m.parameters()):
            assert_equal(param_p.size(),param_m.size())
        m.load_state_dict(p.state_dict())

if __name__ == '__main__':
    # Training settings
    batch_size = 128

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='./mnist',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    model = Net()    
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    trainer = Trainer(model,optimizer,train_loader,test_loader)
    for epoch in range(1, 2):
        scheduler.step()
        trainer.train(epoch)
        trainer.validate()
    model_new = New_Net()
    if torch.cuda.is_available():
        model_new.cuda()
    model_sub_modules_list = [model_new.conv2,model_new.mp]
    pretrained_sub_modules_list = [model.conv2,model.mp]
    load_sub_modules_from_pretrained(pretrained_sub_modules_list,model_sub_modules_list)

    #for param in model.parameters():
    #    param.requires_grad = False
    #params = [{'params':md.parameters(),'lr':0.005} for md in sub_modules]
    #conv1_params = {'params':model.conv1.parameters(),'lr':0.05} 
    #conv2_params = {'params':model.conv2.parameters(),'lr':0.01,'momentum':0.65} 
    #mp_params =  {'params':model.mp.parameters()}
    #fc_parms =  {'params':model.fc.parameters(),'lr':0.002}
    #params = list()
    #params.append(conv1_params)
    #params.append(conv2_params)
    #params.append(mp_params)
    #params.append(fc_parms)
    #optimizer = optim.SGD(params, lr=0.01, momentum=0.5)
    #optimizer = optim.ASGD(model.parameters(), lr=0.001)

   