import _init_paths

import utils
import utils.Tensor as T
import numpy as np
import torch
import os




if __name__=="__main__":

   
    l = list(range(0, 30, 5))
    arr_l=T.to_numpy(l)
    tensor_l = T.to_tensor(arr_l,cuda=False,dtype=torch.FloatTensor)
    var_l = T.to_variable(tensor_l)
    arr_l = T.to_numpy(var_l)
    var_l = T.to_variable(l)
    scale_ = T.to_scalar(var_l)
    shuffle_arr = T.shuffle(arr_l)
    sw_l = T.slide_window_(arr_l,3)
    sw_2 = T.slide_window_(arr_l,4,2)

    print('finished')