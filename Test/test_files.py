import _init_paths

import os
import utils
from utils.files import *



if __name__=="__main__":

   
    file_name='.\\testData\\train.tsv'
    file_path_name = os.path.abspath(file_name)

    if F_file_exists(file_path_name) :
        print('file existing')
        name = F_get_filename_from_fpath(file_path_name) #返回文件名
        print(name)

    dir_name = os.path.dirname('./testData/')
    path_name = os.path.abspath(dir_name)
    print('including sub dir....')
    p,f=F_get_filepaths_and_filenames(path_name) #显示该文件目录及子目录所有文件
    for path_,file_ in zip(p,f):
        print(path_,file_)

    print('get images files....')
    p,f=F_get_filepaths_and_filenames(path_name,file_ext="jpg") #显示该文件目录及子目录所有文件
    for path_,file_ in zip(p,f):
        print(path_,file_)
   
    
    print('get files....')
    f=F_list_files(path_name) #显示该文件目录及子目录所有文件
    for file_ in f:
        print(file_)

    print('including sub dir....')
    f=F_subdir_file_list(path_name) #显示该文件目录及子目录所有文件
    for file_ in f:
        print(file_)




    print('finished')