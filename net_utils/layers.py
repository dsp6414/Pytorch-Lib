import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import random

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Lambda(nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return x.view(x.shape[0], -1)

def get_features(x):
      
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class BWtoRGB(nn.Module):
    def __init__(self):
        super(BWtoRGB, self).__init__()

    def forward(self, x):
        assert len(list(x.size())) == 4
        chans = x.size(1)
        if chans < 3:
            return torch.cat([x, x, x], 1)
        else:
            return x


class Layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.
        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(Layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

class GatedConv2d(nn.Module):
    '''from jmtomczak's github '''
    def __init__(self, input_channels, output_channels, kernel_size,
                 stride, padding=0, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

class GatedConvTranspose2d(nn.Module):
    ''' from jmtomczak's github'''
    def __init__(self, input_channels, output_channels, kernel_size,
                 stride, padding=0, dilation=1, activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride,
                 dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1,
                                          stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x

def str_to_activ_module(str_activ):
    ''' Helper to return a tf activation given a str'''
    str_activ = str_activ.strip().lower()
    activ_map = {
        'identity': Identity,
        'elu': nn.ELU,
        'sigmoid': nn.Sigmoid,
        'log_sigmoid': nn.LogSigmoid,
        'tanh': nn.Tanh,
        'softmax': nn.Softmax,
        'log_softmax': nn.LogSoftmax,
        'selu': nn.SELU,
        'relu': nn.ReLU,
        'softplus': nn.Softplus,
        'hardtanh': nn.Hardtanh,
        'leaky_relu': nn.LeakyReLU,
        'softsign': nn.Softsign
    }

    assert str_activ in activ_map, "unknown activation requested"
    return activ_map[str_activ]


def Conv_Relu(in_channels, out_channels, kernel_size=3, stride=1,
              padding=1, bias=True):
    l = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*l)

def Conv_BN_Relu(in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=False):
    l= [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*l)

def Linear_BN_Relu(in_channels, out_channels, dropout=None, bias=False):
    layers = [
        nn.Linear(in_channels, out_channels, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def Conv_1x1_BN(inp, oup):
    layers=[
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*layers)


def Up_Pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ChannelAttention(nn.Module):

    def __init__(self, inplanes, reduction_ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Output size of 1x1xC
        self.fc = nn.Sequential(
            nn.Linear(inplanes, inplanes // reduction_ratio),
            nn.ReLU(),
            nn.Linear(inplanes // reduction_ratio, inplanes),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, num_channels)
        y = self.fc(y).view(batch_size, num_channels, 1, 1)
        return x * y


class Pooling_cat(nn.Module):
    def __init__(self, sz=(1, 1)):
        super().__init__()
        sz = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return nn.functional.avg_pool2d(
            input=x,
            kernel_size=(h, w))


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return nn.functional.max_pool2d(
            input=x,
            kernel_size=(h, w))


class GlobalConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], 1)


def Layer_Classifier(in_feat, n_classes, activation=None, p=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=n_classes)
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


def MLP_Classifier(in_feat, hidden_feat, n_classes, activation=None, p=0.01, p2=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=hidden_feat),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=hidden_feat),
        nn.Dropout(p2),
        nn.Linear(in_features=hidden_feat, out_features=n_classes)       
    ]

    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

def init_weights(module,init_func=torch.nn.init.normal,const=0.0):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #print("initializing ", m, " with xavier init")
            init_func(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                #print("initial bias from ", m, " with zeros")
                nn.init.constant(m.bias, const)
        elif isinstance(m, nn.Sequential):
            for mod in m:
                init_weights(mod,init_func)

    return module

def set_unfreeze_layers_params(model,freeze_modules,lr=1e-4,momentum =0,weight_decay=0): #用于微调
    params=[]
    for mod in model.children():
        if mod not in freeze_modules:
            tmp = {'params':mod.parameters(),'lr':lr,'momentum':momentum,'weight_decay':weight_decay}
            params.append(tmp)
    return params

def freeze_layers(model,freezen_modules,lr=1e-4,momentum =0.65,weight_decay=1e-6): 
    params=[]
    for mod in freezen_modules:
        for param in mod.parameters():
            param.requires_grad=False
    for mod in model.modules():
        for param in mod.parameters():
            if param.requires_grad: 
                tmp = {'params':mod.parameters(),'lr':lr,'momentum':momentum,'weight_decay':weight_decay}
                params.append(tmp)
    return params

def set_freezen_modules_require_grad_false(model,freezen_modules):
    for mod in freezen_modules:
        for param in mod.parameters():                             
            param.requires_grad=False

def resume_freezen_modules_require_grad(model,freezen_modules):
    for mod in freezen_modules:
        for param in mod.parameters():
            param.requires_grad=True


def set_module_params(module,lr=1e-4,momentum =0,weight_decay=0): #用于微调

    params = {'params':module.parameters(),'lr':lr,'momentum':momentum,'weight_decay':weight_decay}

    return params


def cut_model(model, cut):
    return nn.Sequential(*list(model.children())[:cut])



import warnings


def clip_grad_norm_(optimizer, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor]): an iterable of Tensors that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = [param for group in optimizer.param_groups for param in group['params'] if param.grad is not None ]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def clip_grad_norm(optimizer, max_norm=0.1, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
    warnings.warn("torch.nn.utils.clip_grad_norm is now deprecated in favor "
                  "of torch.nn.utils.clip_grad_norm_.", stacklevel=2)
    return clip_grad_norm_(optimizer, max_norm, norm_type)


def clip_grad_value_(optimizer, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor]): an iterable of Tensors that will have
            gradients normalized
        clip_value (float or int): maximum allowed value of the gradients
            The gradients are clipped in the range [-clip_value, clip_value]
    """
    clip_value = float(clip_value)
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                p.grad.data.clamp_(min=-clip_value, max=clip_value)


def get_requires_grad_params(model):   
    model_params = filter(lambda p: p.requires_grad, model.parameters())   
    return model_params

class PCA(nn.Module):
    def __init__(self, pca_model, scaler_model=None):
        super(PCA, self).__init__()
        
        if scaler_model is not None:
            self.has_scale = True
            self.scale = nn.Parameter(torch.from_numpy(scaler_model.scale_).type(torch.FloatTensor))
            self.scale_mean = nn.Parameter(torch.from_numpy(scaler_model.mean_).type(torch.FloatTensor))
        else:
            self.has_scale = False
        
        if pca_model.mean_ is None:
            self.mean = nn.Parameter(torch.zeros(1))
        else:
            self.mean = nn.Parameter(torch.from_numpy(pca_model.mean_).type(torch.FloatTensor))
        self.components = nn.Parameter(torch.from_numpy(pca_model.components_).t().type(torch.FloatTensor))
        self.whiten = pca_model.whiten
        self.explained_variance = nn.Parameter(torch.from_numpy(pca_model.explained_variance_).type(torch.FloatTensor))

        self.n_components = pca_model.n_components_
        self.noise_variance = pca_model.noise_variance_
        self.singular_values = torch.from_numpy(pca_model.singular_values_).type(torch.FloatTensor)
        self.explained_variance_ratio = torch.from_numpy(pca_model.explained_variance_ratio_).type(torch.FloatTensor)
    
    def forward(self, x):
        if self.has_scale:
            x = x - self.scale_mean
            x = x / self.scale
        
        if self.mean is not None:
            x = x - self.mean
        
        x_transformed = torch.mm(x, self.components)
        
        if self.whiten:
            x_transformed = x_transformed / torch.sqrt(self.explained_variance)            
        
        return x_transformed




