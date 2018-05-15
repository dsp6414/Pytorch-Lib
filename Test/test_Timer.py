import _init_paths

import utils
import utils.Timer as Tm
import numpy as np
import torch
import os




if __name__=="__main__":

   
    timer = Tm.Timer()
    timer.tic()
    for i in range(1,10000):
        continue
    t=timer.toc(average=False)
    print(t)

    print('finished')