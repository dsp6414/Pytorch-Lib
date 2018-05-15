import _init_paths

import os

import utils
from utils.Tensor import to_numpy
from utils.files import F_file_exists
from utils.imgs import *

import torch
import torchvision
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #(None,20,20,20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x_tensor1 = x.detach()
        x = F.relu(self.mp(self.conv1(x)))
        x_tensor2 = x.detach()
        x = F.relu(self.mp(self.conv2(x)))        
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return x,x_tensor1,x_tensor2

if __name__=="__main__":

   
    file_name='.\\testData\\cat.10051.jpg'
    file_path_name = os.path.abspath(file_name)
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

    if F_file_exists(file_path_name) and is_image_file(file_path_name,IMG_EXTENSIONS):
        img=load_img_as_arr(file_path_name)
        plot_img_arr(img)
        print('plot from path')
        plot_img_from_fpath(file_path_name)
        arr = img_to_array(img)
        plot_img_arr(arr)
        img = array_to_img(arr)
        plot_img(img)
    

    batch_size = 1

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)
 

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = Net()    

    for data,label in train_loader:
        break
    data,label = Variable(data), Variable(label)
    _,tensor1,tensor2 = model(data)
    
    plot_img_4D_tensor(tensor1.data,nrow=6,padding=0,scale_each=True,normalize=True)
    plot_img_4D_tensor(tensor2.data,nrow=6,padding=0,scale_each=True,normalize=True)
    plot_img_4D_tensor(model.conv1.weight.data,nrow=6,padding=0,scale_each=True,normalize=True)


    print('finished')