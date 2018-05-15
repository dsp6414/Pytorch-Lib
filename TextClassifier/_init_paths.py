"""Add {PROJECT_ROOT}/lib. to PYTHONPATH

Usage:
import this module before import any modules under lib/
e.g 
    import _init_paths
    from core.config import cfg
""" 

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.abspath(osp.dirname(__file__))
parent_dir= osp.abspath(osp.dirname(osp.dirname(__file__)))


# Add lib to PYTHONPATH
utils_path = osp.join(parent_dir, 'utils')
add_path(utils_path)
net_utils_path = osp.join(parent_dir, 'net_utils')
add_path(net_utils_path)
losses_path = osp.join(parent_dir, 'losses')
add_path(losses_path)
models_path = osp.join(parent_dir, 'models')
add_path(models_path)
datasets_path = osp.join(parent_dir, 'datasets')
add_path(datasets_path)
optim_path = osp.join(parent_dir, 'optim')
add_path(optim_path)
data_loader_path = osp.join(parent_dir, 'data_loader')
add_path(data_loader_path)


add_path(parent_dir)
add_path(this_dir)

#print(sys.path)

