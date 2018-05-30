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
from data_loader.cifar_loader import cifar_loader

import utils
from utils.metric import AccuracyMeter,AverageMeter,get_accuracy
import net_utils
from net_utils.layers import Flatten,set_unfreeze_layers_params,set_module_params,clip_grad_norm
from net_utils.layers import Layer_Classifier
import checkpoint
from checkpoint.CheckPoints import CheckPoints
from torchvision.models.resnet import resnet34
from scipy.stats.distributions import norm,uniform
from utils.misc import *
import Template
from Template.cifar_10_models.resnet import ResNet34


class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def arg_parser():
    parser = argparse.ArgumentParser(description='mnist')
    parser.add_argument('--lr', type=float, default=0.0001,
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

    parser.add_argument('--data', type=str, default='./data/Template/cifar-10',
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
        self.args =args
        

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
            clip_grad_norm(self.optimizer, max_norm=1) #防止梯度爆炸
            self.optimizer.step()

            train_loss.update(float(loss.data), x.size(0))

            y_pred = output.data.max(dim=1)[1]
            #correct = int(y_pred.eq(y.data).cpu().sum())
            _,correct,_=get_accuracy(y.data,y_pred)
            train_acc.update(correct, x.size(0))
            if i % 100 ==0:
                print('\nTrain Epoch/batch| [{}/{}]: Average batch loss:{:.6f},acc: {:.6f}\n'.format(epoch,i,train_loss.average,train_acc.accuracy))

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
        print('\nTest Epoch [{}]: Average batch loss:{:.6f},acc: {:.6f}\n'.format(epoch,valid_loss.average,valid_acc.accuracy))
        return valid_loss.average, valid_acc.accuracy

    def print_msg(self,proc,epoch,loss,acc):
        print('proc={},epoch={}:loss={},acc={}'.format(proc,epoch,loss,acc))

def main():
    print('Programming starting ...')
    
    args= arg_parser()
    torch.manual_seed(args.seed)
    train_loader,test_loader = cifar_loader(root=args.data,train_batch_size=args.batch_size,
                 valid_batch_size=args.batch_size,
                 train_shuffle=True,
                 valid_shuffle=False)

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    
    model = ResNet34()
    checkpointer=CheckPoints(model,'./data/checkpoint')
    checkpointer.load_checkpoint_from_filename('model-best.chkpt')
    if args.use_cuda:
        model.cuda()
    
    
    n_iters=200
    train_loss_epochs=np.zeros((n_iters,))
    train_acc_epochs=np.zeros((n_iters,))
    test_loss_epochs=np.zeros((n_iters,))
    test_acc_epochs=np.zeros((n_iters,))

    #Adam算法跑基本模型
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    trainer = Trainer(model,optimizer,train_loader,test_loader,args=args) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    epochs=50
    i=0
    for epoch in np.arange(epochs+1):      
        scheduler.step()
        train_loss, train_acc=trainer.train(i)
        train_loss_epochs[i]=train_loss
        train_acc_epochs[i]=train_acc
        trainer.print_msg('train',i,train_loss,train_acc)
        test_loss, test_acc=trainer.validate(i)
        test_loss_epochs[i]=test_loss
        test_acc_epochs[i]=test_acc
        trainer.print_msg('test',i,test_loss,test_acc)
        is_best=checkpointer.save_checkpoint(i,train_loss,train_acc,test_loss,test_acc,save_best=True)
        #if not is_best:
        #    checkpointer.load_checkpoint_from_filename('model-best.chkpt')
        #i+=1


    #逐层微调
    layers = len(list(model.children()))
    model_list = nn.ModuleList(list(model.children()))

    epochs = 20
    args.lr = 1e-5
    for k in np.arange(layers):
        mod = model_list[k]
        optimizer = torch.optim.SGD(mod.parameters(), lr=args.lr,weight_decay=0,momentum=0)
        trainer = Trainer(model,optimizer,train_loader,test_loader,args=args) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        for epoch in np.arange(epochs+1):      
            scheduler.step()
            train_loss, train_acc=trainer.train(i)
            train_loss_epochs[i]=train_loss
            train_acc_epochs[i]=train_acc
            trainer.print_msg('train',i,train_loss,train_acc)
            test_loss, test_acc=trainer.validate(i)
            test_loss_epochs[i]=test_loss
            test_acc_epochs[i]=test_acc
            trainer.print_msg('test',i,test_loss,test_acc)
            is_best=checkpointer.save_checkpoint(i,train_loss,train_acc,test_loss,test_acc,save_best=True)
            #if not is_best:
            #    checkpointer.load_checkpoint_from_filename('model-best.chkpt')
            i+=1

    #网格随机搜索调参
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    trainer = Trainer(model,optimizer,train_loader,test_loader,args=args)
    

    param_grid = {'lr':[1e-4,1e-5,1e-6,1e-7],'momentum':uniform(0.6,0.25),'weight_decay':[1e-3,1e-5,1e-7]}

    n_iters=100
    param_list = list(ParameterSampler(param_grid, n_iter=n_iters))

    for parm in param_list:
        args.lr=parm['lr']
        args.momentum=parm['momentum']
        args.weight_decay=parm['weight_decay'] 
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)
        trainer = Trainer(model,optimizer,train_loader,test_loader,args=args)
       
        #scheduler.step()
        train_loss, train_acc=trainer.train(i)
        train_loss_epochs[i]=train_loss
        train_acc_epochs[i]=train_acc
        trainer.print_msg('train',i,train_loss,train_acc)
        test_loss, test_acc=trainer.validate(i)
        test_loss_epochs[i]=test_loss
        test_acc_epochs[i]=test_acc
        trainer.print_msg('test',i,test_loss,test_acc)
        is_best=checkpointer.save_checkpoint(i,train_loss,train_acc,test_loss,test_acc,save_best=True)
        if not is_best:
            checkpointer.load_checkpoint_from_filename('model-best.chkpt')
        i+=1

    #调整指定的层的参数
    params=[]
    param=set_module_params(model.linear,lr=1e-5) #设置要调整的层，加入到参数列表
    params.append(param)
    
    optimizer = torch.optim.SGD(params, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    trainer = Trainer(model,optimizer,train_loader,test_loader,args=args)
    epochs=20

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
        checkpointer.save_checkpoint(i,train_loss,train_acc,test_loss,test_acc,save_best=True)
        i+=1


    data ={'train_loss':train_loss_epochs,'train_acc':train_acc_epochs,'test_loss':test_loss_epochs,'test_acc':test_acc_epochs}
    torch.save(data,'./data/Template/data.pt')
    print('Finished...')

if __name__=="__main__":
    main()
