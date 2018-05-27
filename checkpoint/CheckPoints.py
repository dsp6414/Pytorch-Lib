import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import random
import os
import collections
from collections import OrderedDict

class CheckPoints():

    def __init__(self, net, directory=None):
        self.net = net
        if directory:
            self.__ensure_directory_exists__(directory)
            self.storage_directory = directory
        else:
            self.storage_directory = os.path.join(".", "tmp/checkpoint")    
        self.best_acc=0.0


    def __ensure_directory_exists__(self, directory):
        if os.path.exists(directory):
            return

        os.makedirs(directory)

    def save_checkpoint(self,epoch,train_loss,train_acc,test_loss,test_acc,save_best=True):
        is_best=False
        print("saving model...epoch={}".format(epoch))
        checkpoint = {}
        checkpoint["state_dict"] = self.net.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["train_loss"] = train_loss
        checkpoint["train_acc"] = train_acc
        checkpoint["test_loss"] = test_loss
        checkpoint["test_acc"] = test_acc
        #torch.save(checkpoint,  os.path.join(self.storage_directory, "model-epoch-{}.chkpt".format(epoch)))
        if save_best and test_acc>self.best_acc:
            torch.save(checkpoint,  os.path.join(self.storage_directory, "model-best.chkpt"))
            self.best_acc = test_acc
            is_best=True
        return is_best

    def load_checkpoint(self,epoch):
        print("loading model...epoch={}".format(epoch))
        checkpoint_path_name =  os.path.join(self.storage_directory, "model-epoch-{}.chkpt".format(epoch))
        checkpoint = torch.load(checkpoint_path_name)
        self.net.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        train_loss= checkpoint["train_loss"]  
        train_acc = checkpoint["train_acc"]  
        test_loss = checkpoint["test_loss"]  
        test_acc = checkpoint["test_acc"] 
        self.best_acc = test_acc
        return epoch,train_loss,train_acc,test_loss,test_acc

    def load_checkpoint_from_filename(self,file_name):
        print("loading model from file_name...{}".format(file_name))
        checkpoint_path_name =  os.path.join(self.storage_directory, file_name)
        checkpoint = torch.load(checkpoint_path_name)
        self.net.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        train_loss= checkpoint["train_loss"]  
        train_acc = checkpoint["train_acc"]  
        test_loss = checkpoint["test_loss"]  
        test_acc = checkpoint["test_acc"] 
        self.best_acc = test_acc
        return epoch,train_loss,train_acc,test_loss,test_acc


    def _get_filepaths_and_filenames(root, file_ext=None, topdown=True,sort=True, strip_ext=False):
        filepaths = []
        fnames = []
        for (dirpath, dirnames, filenames) in os.walk(root,topdown=topdown):
            filepaths.extend(os.path.join(dirpath, f) 
                for f in filenames if file_ext is None or any(f.endswith(extension) for extension in file_ext))
            fnames.extend([f for f in filenames if file_ext is None or any(f.endswith(extension) for extension in file_ext)])
        if strip_ext:
            fnames = [os.path.splitext(f)[0] for f in fnames]
        if sort:
            return sorted(filepaths), sorted(fnames)
        return filepaths, fnames



    def load_state_dict_from_source(self, source_state):  #加载权值
        new_dict = OrderedDict()
        for k, v in self.net.state_dict().items():
            if k in source_state and v.size() == source_state[k].size():
                new_dict[k] = source_state[k]
            else:
                new_dict[k] = v
        self.net.load_state_dict(new_dict)


    def load_sub_modules_from_pretrained(self,pretraind_sub_modules_list,model_sub_modules_list):
        for p,m in zip(pretraind_sub_modules_list,model_sub_modules_list):
            for param_p,param_m in zip(p.parameters(),m.parameters()):
                assert_equal(param_p.size(),param_m.size())
            m.load_state_dict(p.state_dict())




