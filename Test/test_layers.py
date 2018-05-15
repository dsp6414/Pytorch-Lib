import _init_paths

import os
import net_utils
from net_utils.layers import *
import torch
from torch.autograd import Variable
from torch.optim import SGD

    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #(None,20,20,20)
        self.mp = nn.MaxPool2d(2)
        self.fc = Layer_Classifier(320,10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return x


if __name__=="__main__":

    #x = torch.randn(2,3,4)
    #flatten = Flatten()
    #y=flatten(x)
    #print(y)
   
    #f = lambda x:torch.sum(x)
    #lambda_sum = Lambda(f)
    #y=lambda_sum(x)
    #print(y)

    #x = Variable(torch.randn(2,1,28,28))
    #conv_relu = Conv_Relu(1,5)
    #y=conv_relu(x)
    #print(y)

    #x = Variable(torch.randn(2,1,28,28))
    #conv_bn_relu = Conv_BN_Relu(1,5)
    #y=conv_bn_relu(x)
    #print(y)

    #x = Variable(torch.randn(2,1,28,28))
    #features = get_features(x)
    #x = flatten(x)
    #linear_bn_relu = Linear_BN_Relu(features,10,dropout=0.5)
    #y=linear_bn_relu(x)
    #print(y)

    #x = Variable(torch.randn(2,1,28,28))
    #conv_1x1_bn = Conv_1x1_BN(1,5)
    #y=conv_1x1_bn(x)
    #print(y)

    #x = Variable(torch.randn(2,3,7,7))
    #globe_avg_pool = GlobalAvgPool2d()
    #y = globe_avg_pool(x)
    #print(y)

    #x = Variable(torch.randn(2,3,7,7))
    #globe_max_pool = GlobalMaxPool2d()
    #y = globe_max_pool(x)
    #print(y)

    #x = Variable(torch.randn(2,150))
    #classifier = Layer_Classifier(150,10)
    #y = classifier(x)
    #print(y)

    #x = Variable(torch.randn(2,150))
    #classifier = MLP_Classifier(150,50,10)
    #y = classifier(x)

    x = Variable(torch.randn(5,1,28,28))
    model = Net()
    #func = torch.nn.init.xavier_normal
    ##init_weights(model,func)
    #init_weights(model)

    #clip_gradient_norm(model,0.1)

    for name,mod in model.named_modules(): 
        print(name,mod)
   
    #freeze_layers = [model.conv1,model.conv2]
    #unfreeze_layers_params = set_unfreeze_layers_params(model,freeze_layers)
    #optim = torch.optim.SGD(unfreeze_layers_params,lr=0.01)

    
    #params=[]
    #param=set_module_params(model.conv2,lr=0.0015)
    #params.append(param)
    #param=set_module_params(model.fc,lr=0.002,momentum=0.65,weight_decay=1e-6)
    #params.append(param)
    #optim = torch.optim.SGD(params,lr=0.01)
    #clip_gradient_optim(optim,0.01)
    
    
    

    print('finished')