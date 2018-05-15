import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F
from torch.autograd import Variable
import re
from functools import partial
import matplotlib.pyplot as plt
from six import iterkeys, iteritems, u, string_types, unichr

if sys.version_info[0] >= 3:
    unicode = str

from smart_open import *
try:

    from PIL import ImageEnhance

    from PIL import Image as pil_image

except ImportError:

    pil_image = None

if pil_image is not None:

    _PIL_INTERPOLATION_METHODS = {

        'nearest': pil_image.NEAREST,

        'bilinear': pil_image.BILINEAR,

        'bicubic': pil_image.BICUBIC,

    }

    # These methods were only introduced in version 3.4.0 (2016).

    if hasattr(pil_image, 'HAMMING'):

        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING

    if hasattr(pil_image, 'BOX'):

        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX

    # This method is new in version 1.1.3 (2013).

    if hasattr(pil_image, 'LANCZOS'):

        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

def expand_dims(tensor, dim=0):
    shape = list(tensor.size())
    shape.insert(dim, 1)
    return tensor.view(*shape)


def squeeze_expand_dim(tensor, axis):
    ''' helper to squeeze a multi-dim tensor and then
        unsqueeze the axis dimension if dims < 4'''
    tensor = torch.squeeze(tensor)
    if len(list(tensor.size())) < 4:
        return tensor.unsqueeze(axis)

    return tensor

def append_to_csv(data, filename):
    with open(filename, 'ab') as f:
        np.savetxt(f, data, delimiter=",")


def is_cuda(tensor_or_var):
    tensor = to_data(tensor_or_var)
    cuda_map = {
        torch.cuda.FloatTensor: True,
        torch.FloatTensor: False,
        torch.cuda.IntTensor: True,
        torch.IntTensor: False,
        torch.cuda.LongTensor: True,
        torch.LongTensor: False,
        torch.cuda.HalfTensor: True,
        torch.HalfTensor: False,
        torch.cuda.DoubleTensor: True,
        torch.DoubleTensor: False
    }
    return cuda_map[type(tensor)]


def to_categorical(y, num_classes=None):

    """Converts a class vector (integers) to binary class matrix.



    E.g. for use with categorical_crossentropy.



    # Arguments

        y: class vector to be converted into a matrix

            (integers from 0 to num_classes).

        num_classes: total number of classes.



    # Returns

        A binary matrix representation of the input.

    """

    y = np.array(y, dtype='int')

    input_shape = y.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:

        input_shape = tuple(input_shape[:-1])

    y = y.ravel()

    if not num_classes:

        num_classes = np.max(y) + 1

    n = y.shape[0]

    categorical = np.zeros((n, num_classes), dtype=np.float32)

    categorical[np.arange(n), y] = 1

    output_shape = input_shape + (num_classes,)

    categorical = np.reshape(categorical, output_shape)

    return categorical

def normalize(x, axis=-1, order=2):

    """Normalizes a Numpy array.



    # Arguments

        x: Numpy array to normalize.

        axis: axis along which to normalize.

        order: Normalization order (e.g. 2 for L2 norm).



    # Returns

        A normalized copy of the array.

    """

    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))

    l2[l2 == 0] = 1

    return x / np.expand_dims(l2, axis)

def array_to_img(x, data_format=None, scale=True):

    """Converts a 3D Numpy array to a PIL Image instance.



    # Arguments

        x: Input Numpy array.

        data_format: Image data format.

        scale: Whether to rescale image values

            to be within [0, 255].



    # Returns

        A PIL Image instance.



    # Raises

        ImportError: if PIL is not available.

        ValueError: if invalid `x` or `data_format` is passed.

    """

    if pil_image is None:

        raise ImportError('Could not import PIL.Image. '

                          'The use of `array_to_img` requires PIL.')

    x = np.asarray(x, dtype=K.floatx())

    if x.ndim != 3:

        raise ValueError('Expected image array to have rank 3 (single image). '

                         'Got array with shape:', x.shape)



    if data_format is None:

        data_format =  'channels_last'

    if data_format not in {'channels_first', 'channels_last'}:

        raise ValueError('Invalid data_format:', data_format)



    # Original Numpy array x has format (height, width, channel)

    # or (channel, height, width)

    # but target PIL image has format (width, height, channel)

    if data_format == 'channels_first':

        x = x.transpose(1, 2, 0)

    if scale:

        x = x + max(-np.min(x), 0)

        x_max = np.max(x)

        if x_max != 0:

            x /= x_max

        x *= 255

    if x.shape[2] == 3:

        # RGB

        return pil_image.fromarray(x.astype('uint8'), 'RGB')

    elif x.shape[2] == 1:

        # grayscale

        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')

    else:

        raise ValueError('Unsupported channel number: ', x.shape[2])





def img_to_array(img, data_format=None):

    """Converts a PIL Image instance to a Numpy array.



    # Arguments

        img: PIL Image instance.

        data_format: Image data format.



    # Returns

        A 3D Numpy array.



    # Raises

        ValueError: if invalid `img` or `data_format` is passed.

    """

    if data_format is None:

        data_format = 'channels_last'

    if data_format not in {'channels_first', 'channels_last'}:

        raise ValueError('Unknown data_format: ', data_format)

    # Numpy array x has format (height, width, channel)

    # or (channel, height, width)

    # but original PIL image has format (width, height, channel)

    x = np.asarray(img, dtype=np.float32)

    if len(x.shape) == 3:

        if data_format == 'channels_first':

            x = x.transpose(2, 0, 1)

    elif len(x.shape) == 2:

        if data_format == 'channels_first':

            x = x.reshape((1, x.shape[0], x.shape[1]))

        else:

            x = x.reshape((x.shape[0], x.shape[1], 1))

    else:

        raise ValueError('Unsupported image shape: ', x.shape)

    return x


def load_img(path, grayscale=False, target_size=None,

             interpolation='nearest'):

    """Loads an image into PIL format.



    # Arguments

        path: Path to image file

        grayscale: Boolean, whether to load the image as grayscale.

        target_size: Either `None` (default to original size)

            or tuple of ints `(img_height, img_width)`.

        interpolation: Interpolation method used to resample the image if the

            target size is different from that of the loaded image.

            Supported methods are "nearest", "bilinear", and "bicubic".

            If PIL version 1.1.3 or newer is installed, "lanczos" is also

            supported. If PIL version 3.4.0 or newer is installed, "box" and

            "hamming" are also supported. By default, "nearest" is used.



    # Returns

        A PIL Image instance.



    # Raises

        ImportError: if PIL is not available.

        ValueError: if interpolation method is not supported.

    """

    if pil_image is None:

        raise ImportError('Could not import PIL.Image. '

                          'The use of `array_to_img` requires PIL.')

    img = pil_image.open(path)

    if grayscale:

        if img.mode != 'L':

            img = img.convert('L')

    else:

        if img.mode != 'RGB':

            img = img.convert('RGB')

    if target_size is not None:

        width_height_tuple = (target_size[1], target_size[0])

        if img.size != width_height_tuple:

            if interpolation not in _PIL_INTERPOLATION_METHODS:

                raise ValueError(

                    'Invalid interpolation method {} specified. Supported '

                    'methods are {}'.format(

                        interpolation,

                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))

            resample = _PIL_INTERPOLATION_METHODS[interpolation]

            img = img.resize(width_height_tuple, resample)

    return img

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):

    return [os.path.join(root, f)

            for root, _, files in os.walk(directory) for f in files

            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def to_data(tensor_or_var):
    '''simply returns the data'''
    if isinstance(tensor_or_var, Variable):
        return tensor_or_var.data

    return tensor_or_var

#def to_var(x, volatile=False):
#    if torch.cuda.is_available():
#        x = x.cuda(async=True)
#    return torch.autograd.Variable(x, volatile=volatile)


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



def to_tensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch._TensorBase):
        tensor = data
    if isinstance(data, torch.autograd.Variable):
        tensor = data.data
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


def file_exists(filename):
    return os.path.isfile(filename)

def delete_file(filename):
    if file_exists(filename):
        os.remove(filename)


def is_dataset(x):
    return isinstance(x, torch.utils.data.Dataset)

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


def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def one_hot(dim, i):
    t = torch.Tensor(dim).zero_()
    t.narrow(0, i, 1).fill_(1)
    return to_var(t)

def slide_window_(a, kernel, stride=None):
    """Expands last dimension to help compute sliding windows.
    
    Args:
        a (Tensor or Array): The Tensor or Array to view as a sliding window.
        kernel (int): The size of the sliding window.
        stride (tuple or int, optional): Strides for viewing the expanded dimension (default 1)
    The new dimension is added at the end of the Tensor or Array.
    Returns:
        The expanded Tensor or Array.
    Running Sum Example::
        >>> a = torch.Tensor([1, 2, 3, 4, 5, 6])
         1
         2
         3
         4
         5
         6
        [torch.FloatTensor of size 6]
        >>> a_slided = dlt.util.slide_window_(a.clone(), kernel=3, stride=1)
         1  2  3
         2  3  4
         3  4  5
         4  5  6
        [torch.FloatTensor of size 4x3]
        >>> running_total = (a_slided*torch.Tensor([1,1,1])).sum(-1)
          6
          9
         12
         15
        [torch.FloatTensor of size 4]
    Averaging Example::
        >>> a = torch.Tensor([1, 2, 3, 4, 5, 6])
         1
         2
         3
         4
         5
         6
        [torch.FloatTensor of size 6]
        >>> a_sub_slide = dlt.util.slide_window_(a.clone(), kernel=3, stride=3)
         1  2  3
         4  5  6
        [torch.FloatTensor of size 2x3]
        >>> a_sub_avg = (a_sub_slide*torch.Tensor([1,1,1])).sum(-1) / 3.0
         2
         5
        [torch.FloatTensor of size 2]
    """


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

def is_array(x):
    """Checks if input type contains 'array' or 'series' in its typename."""
    return torch.typename(x).find('array') >= 0 or torch.typename(x).find('series') >= 0 

def to_array(x):
    """Converts x to a Numpy Array.
    
    Args:
        x (Variable, Tensor or Array): Input to be converted. Can also be on the GPU.
    Automatically gets the data from torch Variables and casts GPU Tensors
    to CPU.
    """
    if is_variable(x):
        x = x.data.clone()
    if is_cuda(x):
        x = x.cpu()
    if is_tensor(x):
        return x.numpy()
    else:
        return x.copy()

def is_variable(x):
    """Checks if input is a Variable instance."""
    return isinstance(x, torch.autograd.Variable)

# This was added to torch in v0.3. Keeping it here too.
def is_tensor(x):
    """Checks if input is a Tensor"""
    return torch.is_tensor(x)
    
def is_cuda(x):
    """Checks if input is a cuda Tensor."""
    return x.is_cuda

from PIL import Image, ImageFilter, ImageChops

def load_image(path):
    image = Image.open(path)
    plt.imshow(image)
    plt.title("Image loaded successfully")
    return image

def deprocess(image):
    return image * torch.Tensor([0.229, 0.224, 0.225])  + torch.Tensor([0.485, 0.456, 0.406])

def file_or_filename(input):
    """Open file with `smart_open`.

    Parameters
    ----------
    input : str or file-like
        Filename or file-like object.

    Returns
    -------
    input : file-like object
        Opened file OR seek out to 0 byte if `input` is already file-like object.

    """
    if isinstance(input, string_types):
        # input was a filename: open as file
        return smart_open(input)
    else:
        # input already a file-like object; just reset to the beginning
        input.seek(0)
        return input

def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert `text` to bytestring in utf8.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).
    encoding : str, optional
        Encoding of `text` for `unicode` function (python2 only).

    Returns
    -------
    str
        Bytestring in utf8.

    """

    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` to unicode.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour, used as parameter for `unicode` function (python2 only).
    encoding : str, optional
        Encoding of `text` for `unicode` function (python2 only).

    Returns
    -------
    str
        Unicode version of `text`.

    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode

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

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)



