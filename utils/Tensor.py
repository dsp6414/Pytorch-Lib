import os
import sys
import torch
from torch.autograd import Variable
import numpy as np
import scipy as sp

from six import iterkeys, iteritems, u, string_types, unichr

if sys.version_info[0] >= 3:
    unicode = str

from smart_open import *

def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    return var.view(-1).data.tolist()[0]

def to_data(tensor_or_var):
    '''simply returns the data'''
    if isinstance(tensor_or_var, Variable):
        return tensor_or_var.data

    return tensor_or_var

def is_array(x):
    """Checks if input type contains 'array' or 'series' in its typename."""
    return torch.typename(x).find('array') >= 0 or torch.typename(x).find('series') >= 0 

def is_torch_data_type(x):
    # pylint: disable=protected-access
    return isinstance(x, (torch.tensor._TensorBase, Variable))

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch._TensorBase):
        return data.cpu().numpy()
    if isinstance(data, torch.autograd.Variable):
        return to_numpy(data.data)


def to_tensor(data, cuda=False,dtype=None):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch._TensorBase):
        tensor = data
    if isinstance(data, torch.autograd.Variable):
        tensor = data.data
    if type is not None:
        tensor.type(dtype)
    if cuda:
        tensor = tensor.cuda()
    return tensor


def to_variable(data):
    if isinstance(data, np.ndarray):
        return to_variable(to_tensor(data))
    if isinstance(data, torch._TensorBase):
        return torch.autograd.Variable(data)
    if isinstance(data, torch.autograd.Variable):
        return data
    else:
        raise ValueError("UnKnow data type: %s, input should be {np.ndarray,Tensor,Variable}" %type(data))

def shuffle(*arrays, **kwargs):

    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)

def slide_window_(a, kernel, stride=None):
   
    if isinstance(kernel, int):
        kernel = (kernel,)
    if stride is None:
        stride = tuple(1 for i in kernel)
    elif isinstance(stride, int):
        stride = (stride,)
    window_dim = len(kernel)
    if is_array(a):
        new_shape = a.shape[:-window_dim] + tuple(int(np.floor((s - kernel[i] )/stride[i]) + 1) for i,s in enumerate(a.shape[-window_dim:])) + kernel
        new_stride = a.strides[:-window_dim] + tuple(s*k for s,k in zip(a.strides[-window_dim:], stride)) + a.strides[-window_dim:]
        return np.lib.stride_tricks.as_strided(a, shape=new_shape, strides=new_stride)
    else:
        new_shape = a.size()[:-window_dim] + tuple(int(np.floor((s - kernel[i] )/stride[i]) + 1) for i,s in enumerate(a.size()[-window_dim:])) + kernel
        new_stride = a.stride()[:-window_dim] + tuple(s*k for s,k in zip(a.stride()[-window_dim:], stride)) + a.stride()[-window_dim:]
        a.set_(a.storage(), storage_offset=0, size=new_shape, stride=new_stride)
        return a

def is_variable(x):
    """Checks if input is a Variable instance."""
    return isinstance(x, torch.autograd.Variable)

def is_tensor(x):
    """Checks if input is a Tensor"""
    return torch.is_tensor(x)
    
def is_cuda(x):
    """Checks if input is a cuda Tensor."""
    return x.is_cuda

def strided_windows(ndarray, window_size):
    """Produce a numpy.ndarray of windows, as from a sliding window.

    Parameters
    ----------
    ndarray : numpy.ndarray
        Input array
    window_size : int
        Sliding window size.

    Returns
    -------
    numpy.ndarray
        Subsequences produced by sliding a window of the given size over the `ndarray`.
        Since this uses striding, the individual arrays are views rather than copies of `ndarray`.
        Changes to one view modifies the others and the original.

    Examples
    --------
    >>> from gensim.utils import strided_windows
    >>> strided_windows(np.arange(5), 2)
    array([[0, 1],
           [1, 2],
           [2, 3],
           [3, 4]])
    >>> strided_windows(np.arange(10), 5)
    array([[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6],
           [3, 4, 5, 6, 7],
           [4, 5, 6, 7, 8],
           [5, 6, 7, 8, 9]])

    """
    ndarray = np.asarray(ndarray)
    if window_size == ndarray.shape[0]:
        return np.array([ndarray])
    elif window_size > ndarray.shape[0]:
        return np.ndarray((0, 0))

    stride = ndarray.strides[0]
    return np.lib.stride_tricks.as_strided(
        ndarray, shape=(ndarray.shape[0] - window_size + 1, window_size),
        strides=(stride, stride))

