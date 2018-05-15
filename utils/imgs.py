import os
import sys
import random
import numpy as np
from skimage import io
from PIL import Image, ImageFilter
from  scipy import ndimage
import scipy.misc
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
import files
import math
irange = range


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

CLASS_COLORS = {
    'green': (0, 128, 0),
    'red': (128, 0, 0),
    'blue': (0, 0, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'grey':(128, 128, 128),
}


if sys.version_info[0] >= 3:
    unicode = str
from smart_open import smart_open
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

def is_image_file(filename,ext): #判断是否是图像文件
    return any(filename.endswith(extension) for extension in ext)


def normalize(x, axis=-1, order=2): #归一化
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)

def array_to_img(x, data_format=None, scale=True): #转换3D numpy array ->PIL Image

    x = np.asarray(x, dtype=np.float32)
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


def img_to_array(img, data_format=None):#转换PIL img-->numpy 

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


def resize_array(x, size): #转换arr尺寸
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array_to_img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype='float32')
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array_to_img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype='float32')
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res

def image_loader(image_name, max_sz=256):
    """ forked from pytorch tutorials """
    r_image = Image.open(image_name)
    mindim = np.min((np.max(r_image.size[:2]), max_sz))

    loader = transforms.Compose([transforms.CenterCrop(mindim),
                                 transforms.ToTensor()])

    image = Variable(loader(r_image))

    return image.unsqueeze(0)


def load_image(path):
    image = Image.open(path)
    return image

def load_img_as_arr(img_path):
    return plt.imread(img_path)


def load_img_as_tensor(img_path): #导入图像转换成tensor
    img_arr = load_img_as_arr(img_path)
    return transforms.ToTensor()(img_arr)


def load_img_as_pil(img_path):
    return Image.open(img_path).convert('RGB')


def save_pil_img(pil_img, fpath):#保存图像
    pil_img.save(fpath)


def save_arr(arr, fpath): #保存arr
    scipy.misc.imsave(fpath, arr)


def norm_meanstd(arr, mean, std): #归一化
    return (arr - mean) / std


def denorm_meanstd(arr, mean, std):
    return (arr * std) + mean


def norm255_tensor(arr): #转float 0-1
    """Given a color image/where max pixel value in each channel is 255
    returns normalized tensor or array with all values between 0 and 1"""
    return arr / 255.


def denorm255_tensor(arr):
    return arr * 255.


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):#显示所有图片文件

    return [os.path.join(root, f)

            for root, _, files in os.walk(directory) for f in files

            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def normalize_feat_map(feat_map): #归一化热图

    return feat_map/np.linalg.norm(feat_map,ord=2,axis=(2),keepdims=True)


def plot_img_arr(arr, fs=(6,6), title=None): #绘制arr 3D
    plt.figure(figsize=fs)
    plt.imshow(arr)
    plt.title(title)
    plt.show()


def plot_heatmaps(tensor,  nrow=8, padding=2,
               normalize=True, range=None, scale_each=False, pad_value=0):
    N,C,H,W = tensor.size()
    tensor = torch.sum(tensor, dim=1)
    grid =torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    plot_img_arr(ndarr)

def plot_img_tensor(tns, fs=(6,6), title=None): #

    arr = tensor_to_arr(tns)
    plot_img_arr(arr, fs, title)


def tensor_to_arr(tns):
    return tns.numpy().transpose((1,2,0))

def img_tensor_to_arr(tns):
    return tns.numpy().transpose((1,2,0))

def img_arr_to_tensor(arr):
    return torch.from_numpy(arr).transpose((2,0,1))



def plot_img_from_fpath(img_path, fs=(8,8), title=None):
    plt.figure(figsize=fs)
    plt.imshow(plt.imread(img_path))
    plt.title(title)
    plt.show()

def plot_img(img, fs=(8,8), title=None):
    plt.figure(figsize=fs)
    plt.imshow(img)
    plt.title(title)
    plt.show()


def plot_meanstd_normed_tensor(tns, mean, std, fs=(6,6), title=None):
    """If normalized with mean/std"""
    tns = denorm255_tensor(tns)
    arr = tns.numpy().transpose((1, 2, 0))
    arr = denorm_meanstd(arr, mean, std)
    plt.figure(figsize=fs)
    plt.imshow(arr)
    if title:
        plt.title(title)
    plt.show()


def get_mean_std_of_dataset(dir_path, sample_size=5):
    fpaths, fnames = files.F_get_filepaths_and_filenames(dir_path)
    random.shuffle(fpaths)
    total_mean = np.array([0.,0.,0.])
    total_std = np.array([0.,0.,0.])
    for f in fpaths[:sample_size]:
        if 'tif' in f:
            img_arr = io.imread(f)
        else:
            img_arr = load_img_as_arr(f)
        mean = np.mean(img_arr, axis=(0,1))
        std = np.std(img_arr, axis=(0,1))
        total_mean += mean
        total_std += std
    avg_mean = total_mean / sample_size
    avg_std = total_std / sample_size
    print("mean: {}".format(avg_mean), "stdev: {}".format(avg_std))
    return avg_mean, avg_std


def plot_binary_mask(arr, threshold=0.5, title=None, color=(255,255,255)):
    arr = format_1D_binary_mask(arr.copy())
    print(arr.shape)
    for i in range(3):
        arr[:,:,i][arr[:,:,i] >= threshold] = color[i]
    arr[arr < threshold] = 0
    plot_img_arr(arr, title=title)


def format_1D_binary_mask(mask):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, 0)
    mask = np.stack([mask,mask,mask],axis=1).squeeze().transpose(1,2,0)
    return mask.astype('float32')


def plot_binary_mask_overlay(mask, img_arr, fs=(18,18), title=None):
    mask = format_1D_binary_mask(mask.copy())
    fig = plt.figure(figsize=fs)
    a = fig.add_subplot(1,2,1)
    a.set_title(title)
    plt.imshow(img_arr.astype('uint8'))
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.show()


def plot_binary_mask_overlay(mask, img_arr, fs=(18,18), title=None):
    mask = format_1D_binary_mask(mask.copy())
    fig = plt.figure(figsize=fs)
    a = fig.add_subplot(1,2,1)
    a.set_title(title)
    plt.imshow(img_arr.astype('uint8'))
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.show()


def plot_samples_from_dir(dir_path, shuffle=False):
    fpaths, fnames = files.get_paths_to_files(dir_path)
    plt.figure(figsize=(16,12))
    start = random.randint(0,len(fpaths)-1) if shuffle else 0
    j = 1
    for idx in range(start, start+6):
        plt.subplot(2,3,j)
        plt.imshow(plt.imread(fpaths[idx]))
        plt.title(fnames[idx])
        plt.axis('off')
        j += 1



def plot_sample_preds_masks(fnames, inputs, preds, fs=(9,9), 
        n_samples=8, shuffle=False):
    start = random.randint(0,len(inputs)-1) if shuffle else 0
    for idx in range(start, start+n_samples):
        print(fnames[idx])
        img = tensor_to_arr(inputs[idx])
        plot_binary_mask_overlay(preds[idx], img, fs, fnames[idx])

def img_make_grids(tensor,nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))
    if tensor.dim() == 4 and tensor.size(1) !=1 and tensor.size(1) !=3 :  
        tensor = torch.sum(tensor, dim=1,keepdim=True)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)



    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


