import _init_paths

import os

import argparse
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import data_loader
from data_loader.mnist_loader import mnist_loader

import utils
from utils.metric import AccuracyMeter,AverageMeter
import net_utils
from net_utils.netutils import save_model_checkpoint
from net_utils.layers import Flatten,set_unfreeze_layers_params,set_module_params,clip_gradient_norm
from net_utils.layers import Layer_Classifier




class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #(None,20,20,20)
        self.mp = nn.MaxPool2d(2)
        #self.fc = nn.Linear(320, 10)
        self.classifer = Layer_Classifier(320,10)
        self.flatten = Flatten()

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        #x = x.view(in_size, -1)  # flatten the tensor
        x = self.flatten(x)
        x = self.classifer(x)
        return x

def arg_parser():
    parser = argparse.ArgumentParser(description='LSTM text classification')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs for train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training [default: 16]')
    parser.add_argument('--step-size', type=int, default=6,
                    help='lr_scheduler step size  [default:5]')
    parser.add_argument('--seed', type=int, default=1721,
                        help='random seed')
    parser.add_argument('--use-cuda', type=bool, default=True,help='enables cuda')

    parser.add_argument('--data', type=str, default='./data/Template/mnist',
                        help='location of the data corpus')
    parser.add_argument('--save', type=str, default='./data/Template/save',
                    help='location of the data corpus')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout (0 = no dropout) [default: 0.5]')
    parser.add_argument('--momentum', type=float, default=0.9,help='momentum parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-7,help='weight_decay parameters')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, valid_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.use_cuda = args.use_cuda
        self.save = args.save
        

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
            clip_gradient_norm(self.model, clip_norm=0.1) #防止梯度爆炸
            self.optimizer.step()

            train_loss.update(float(loss.data), x.size(0))

            y_pred = output.data.max(dim=1)[1]
            correct = int(y_pred.eq(y.data).cpu().sum())
            train_acc.update(correct, x.size(0))
            if i % 100 ==0:
                print('\nTrain Epoch/batch| [{}/{}]: Average batch loss: {:.6f}\n'.format(epoch,i,train_acc.accuracy))

        #save_model_checkpoint(self.model,epoch,self.save)
        return train_loss.average, train_acc.accuracy

    def validate(self,epoch):
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
    def print_msg(self,proc,epoch,loss,acc):
        print('proc={},epoch={}:loss={},acc={}'.format(proc,epoch,loss,acc))

def main():
    print('Programming starting ...')
    
    args= arg_parser()
    torch.manual_seed(args.seed)
    train_loader,test_loader = mnist_loader(root=args.data,train_batch_size=args.batch_size,
                 valid_batch_size=args.batch_size,
                 train_shuffle=True,
                 valid_shuffle=False)
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    model =Net()
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    trainer = Trainer(model,optimizer,train_loader,test_loader,args=args)
    epochs=1
    train_loss_epochs=np.zeros((20,))
    train_acc_epochs=np.zeros((20,))
    test_loss_epochs=np.zeros((20,))
    test_acc_epochs=np.zeros((20,))
    i=0
    for epoch in range(1,epochs+1):
        scheduler.step()
        train_loss, train_acc=trainer.train(i)
        train_loss_epochs[i]=train_loss
        train_acc_epochs[i]=train_acc
        trainer.print_msg('train',i,train_loss,train_acc)
        test_loss, test_acc=trainer.validate(i)
        test_loss_epochs[i]=test_loss
        test_acc_epochs[i]=test_acc
        trainer.print_msg('test',i,test_loss,test_acc)
        i+=1
    
    #固定层，调整剩余的层
    freezen_layers = [model.conv1,model.conv2]
    unfreezen_layers_params = set_unfreeze_layers_params(model,freezen_layers)
    optimizer = torch.optim.SGD(unfreezen_layers_params, lr=0.0001,weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    trainer = Trainer(model,optimizer,train_loader,test_loader,args=args)
    epochs=2

    for epoch in range(1,epochs+1):
        scheduler.step()
        train_loss, train_acc=trainer.train(i)
        train_loss_epochs[i]=train_loss
        train_acc_epochs[i]=train_acc
        trainer.print_msg('train',i,train_loss,train_acc)
        test_loss, test_acc=trainer.validate(i)
        test_loss_epochs[i]=test_loss
        test_acc_epochs[i]=test_acc
        trainer.print_msg('test',i,test_loss,test_acc)
        i+=1
         
    #调整指定的层的参数
    params=[]
    param=set_module_params(model.conv2,lr=0.0015)
    params.append(param)
    param=set_module_params(model.classifer,lr=0.002,momentum=0.65,weight_decay=1e-6)
    params.append(param)
    optimizer = torch.optim.SGD(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    trainer = Trainer(model,optimizer,train_loader,test_loader,args=args)
    epochs=2

    for epoch in range(1,epochs+1):
        scheduler.step()
        train_loss, train_acc=trainer.train(i)
        train_loss_epochs[i]=train_loss
        train_acc_epochs[i]=train_acc
        trainer.print_msg('train',i,train_loss,train_acc)
        test_loss, test_acc=trainer.validate(i)
        test_loss_epochs[i]=test_loss
        test_acc_epochs[i]=test_acc
        trainer.print_msg('test',i,test_loss,test_acc)
        i+=1
    data ={'train_loss':train_loss_epochs,'train_acc':train_acc_epochs,'test_loss':test_loss_epochs,'test_acc':test_acc_epochs}
    torch.save(data,'./data/Template/data.pt')
    print('Finished...')

if __name__=="__main__":
    main()
