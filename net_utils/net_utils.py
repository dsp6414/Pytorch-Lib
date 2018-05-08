import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import random

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

class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.
        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

def conv_relu(in_channels, out_channels, kernel_size=3, stride=1,
              padding=1, bias=True):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
    ]

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, bias=False):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

def linear_bn_relu_drop(in_channels, out_channels, dropout=0.5, bias=False):
    layers = [
        nn.Linear(in_channels, out_channels, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
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

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            SynchronizedBatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
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


def get_fc(in_feat, n_classes, activation=None):
    layers = [
        nn.Linear(in_features=in_feat, out_features=n_classes)
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)


def get_classifier(in_feat, n_classes, activation, p=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=n_classes),
        activation
    ]
    return nn.Sequential(*layers)


def get_mlp_classifier(in_feat, out_feat, n_classes, activation, p=0.01, p2=0.5):
    layers = [
        nn.BatchNorm1d(num_features=in_feat),
        nn.Dropout(p),
        nn.Linear(in_features=in_feat, out_features=out_feat),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=out_feat),
        nn.Dropout(p2),
        nn.Linear(in_features=out_feat, out_features=n_classes),
        activation
    ]
    return nn.Sequential(*layers)

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



def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def save_chkpt(model, chkpt_num, model_dir):

    # Save model
    chkpt_fname = os.path.join(model_dir, "model{}.chkpt".format(chkpt_num))
    torch.save(model.state_dict(), chkpt_fname)

def save_checkpoint(state, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, filename)
    print('Saved model state dict at %s.' % filename)



def load_network_from_chkpt(model, chkpt_num, model_dir):

    chkpt_fname = os.path.join(model_dir, "model{}.chkpt".format(chkpt_num))
    model.load_state_dict(torch.load(chkpt_fname))

    return model

def load_model(fpath, cuda=True):
    if cuda:
        return torch.load(fpath).cuda()
    return torch.load(fpath)


def save_model(model, fpath):
    torch.save(model.cpu(), fpath)


def load_weights_from_source(target, source_state):  #未调试
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            new_dict[k] = v
    target.load_state_dict(new_dict)

def load_weights(model, fpath):
    state = torch.load(fpath)
    model.load_state_dict(state['state_dict'])


def save_weights(model, fpath, epoch=None, name=None):
    torch.save({
        'name': name,
        'epoch': epoch,
        'state_dict': model.state_dict()
    }, fpath)


def freeze_layers(model, n_layers):
    i = 0
    for child in model.children():
        if i >= n_layers:
            break
        print(i, "freezing", child)
        for param in child.parameters():
            param.requires_grad = False
        i += 1


def freeze_nested_layers(model, n_layers):
    i = 0
    for child in model.children():
        for grandchild in child.children():
            if isinstance(grandchild, torch.nn.modules.container.Sequential):
                for greatgrand in grandchild.children():
                    if i >= n_layers:
                        break
                    for param in greatgrand.parameters():
                        param.requires_grad = False
                    print(i, "freezing", greatgrand)
                    i += 1
            else:
                if i >= n_layers:
                    break
                for param in grandchild.parameters():
                    param.requires_grad = False
                print(i, "freezing", grandchild)
                i += 1


def init_nested_layers(module, init_func):
    for child in module.children():
        if len(list(child.children())) > 0:
            init_nested_layers(child, init_func)
        else:
            init_weights(child, init_func)




def init_weights(module,init_func):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print("initializing ", m, " with xavier init")
            init_func(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                print("initial bias from ", m, " with zeros")
                nn.init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for mod in m:
                init_weights(mod,init_func)

    return module


def cut_model(model, cut):
    return nn.Sequential(*list(model.children())[:cut])

def get_requires_grad_params(model):
   
    model_params = filter(lambda p: p.requires_grad, model.parameters())   
    return model_params


def create_criterion(**criterion_params):
    criterion_name = criterion_params.pop('criterion', None)
    if criterion_name is None:
        return None
    criterion = nn.__dict__[criterion_name](**criterion_params)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    return criterion


def create_optimizer(model, **optimizer_params):
    optimizer_name = optimizer_params.pop('optimizer', None)
    if optimizer_name is None:
        return None
    optimizer = torch.optim.__dict__[optimizer_name](
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_params)
    return optimizer

def load_sub_modules_from_pretrained(pretraind_sub_modules_list,model_sub_modules_list):
    for p,m in zip(pretraind_sub_modules_list,model_sub_modules_list):
        for param_p,param_m in zip(p.parameters(),m.parameters()):
            assert_equal(param_p.size(),param_m.size())
        m.load_state_dict(p.state_dict())

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

#def clip_gradient(optimizer, grad_clip):
#    for group in optimizer.param_groups:
#        for param in group['params']:
#            param.grad.data.clamp_(-grad_clip, grad_clip)

#def clip_gradient(optimizer, max_norm, norm_type=2):
#    max_norm = float(max_norm)
#    if norm_type == float('inf'):
#        total_norm = max(p.grad.data.abs().max() for group in optimizer.param_groups for p in group['params'])
#    else:
#        total_norm = 0.0
#        for group in optimizer.param_groups:
#            for p in group['params']:
#                try:
#                    param_norm = p.grad.data.norm(norm_type)
#                    nn = param_norm ** norm_type
#                    # print('norm:', nn, p.grad.size())
#                    total_norm += nn
#                    param_norm ** norm_type
#                except:
#                    pass
#        total_norm = total_norm ** (1. / norm_type)
#    clip_coef = max_norm / (total_norm + 1e-6)
#    if clip_coef < 1:
#        for group in optimizer.param_groups:
#            for p in group['params']:
#                try:
#                    p.grad.data.mul_(clip_coef)
#                except:
#                    pass
#    return total_norm

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def average_checkpoints(inputs): #权值平均
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model'] #注意model关键字key

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            if k not in params_dict:
                params_dict[k] = []
            params_dict[k].append(model_params[k])

    averaged_params = collections.OrderedDict()
    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        summed_v = None
        for x in v:
            summed_v = summed_v + x if summed_v is not None else x
        averaged_params[k] = summed_v / len(v)
    new_state['model'] = averaged_params
    return new_state




