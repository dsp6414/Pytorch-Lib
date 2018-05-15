# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import _init_paths

import utils
from utils.Tensor import to_numpy,to_variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#from ASGD import ASGD
import matplotlib.pyplot as plt
import numpy as np

from utils.metric import AccuracyMeter,MovingAverageMeter,AverageMeter




def adjust_learning_rate(optimizer,epoch,adjust_epochs):
    """Sets the learning rate """     
    for param_group in optimizer.param_groups:
        lr = param_group['lr']*(0.1 ** (epoch // adjust_epochs))
        param_group['lr'] = lr

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

if __name__ == '__main__':
    # Training settings
    batch_size = 128

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)
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
    model =Net()
    if torch.cuda.is_available():
        model.cuda()

    #sub_modules = [model.conv1,model.conv2,model.mp,model.fc]
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    trainer = Trainer(model,optimizer,train_loader,test_loader,use_cuda=True)
    for epoch in range(1, 5):
        scheduler.step()
        trainer.train(epoch)
        trainer.validate()

   