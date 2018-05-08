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

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=None):
  if labels is None: labels = range(cm.shape[0])
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(cm.shape[0])
  plt.xticks(tick_marks, labels, rotation=45)
  plt.yticks(tick_marks, labels)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


def normalize_(x, axis=-1, order=2):
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


if __name__ == '__main__':
   

    print('finished')


