import os
import torch.utils.data
from torch.utils.data import DataLoader



def get_batch(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    inputs, targets, img_paths = next(iter(dataloader))
    return inputs, targets, img_paths

