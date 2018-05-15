import _init_paths




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import utils
from utils.metric import AccuracyMeter, AverageMeter, MovingAverageMeter
import torchvision
from torchvision import datasets, transforms

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, valid_loader, use_cuda=False):
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

        return train_loss.average, train_acc.accuracy

    def validate(self):
        self.model.eval()

        valid_loss = AverageMeter()
        valid_acc = AccuracyMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x = Variable(x, volatile=True)
            y = Variable(y)

            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            valid_loss.update(float(loss.data), x.size(0))

            y_pred = output.data.max(dim=1)[1]
            correct = int(y_pred.eq(y.data).cpu().sum())
            valid_acc.update(correct, x.size(0))

        return valid_loss.average, valid_acc.accuracy


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

def print_msg(proc,epoch,loss,acc):
    print('proc={},epoch={}:loss={},acc={}'.format(proc,epoch,loss,acc))

if __name__=="__main__":
    
    print('start....')
    # Training settings
    batch_size = 32

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
    model = Net()    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model=model,optimizer=optimizer,train_loader=train_loader,valid_loader=test_loader,use_cuda=True)
    epochs = 30
    for epoch in range(1, epochs+1):
        train_loss, train_acc=trainer.train(epoch)
        print_msg('train',epoch,train_loss,train_acc)
        test_loss, test_acc=trainer.validate()
        print_msg('test',epoch,test_loss,test_acc)
  

    print('finished')