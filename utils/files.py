import os
import random
from glob import glob
import shutil
import gzip
import pickle
import json
from contextlib import closing
from zipfile import ZipFile, ZIP_DEFLATED
import re
from smart_open import smart_open

def F_file_exists(fname): #文件是否存在
    return os.path.exists(fname)

def F_file_or_filename(input): #打开文件

    if isinstance(input, str):
        # input was a filename: open as file
        return smart_open(input)
    else:
        # input already a file-like object; just reset to the beginning
        input.seek(0)
        return input


def F_get_filename_from_fpath(fpath): #从全路径中得到文件名
    return os.path.basename(fpath)



def F_get_filepaths_and_filenames(root, file_ext=None, topdown=True,sort=True, strip_ext=False): #得到给定目录包括子目录下，文件全路径和文件名
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


def F_save_json(fpath, dict_): #保存为Json格式
    with open(fpath, 'w') as f:
        json.dump(dict_, f, indent=4, ensure_ascii=False)


def F_load_json(fpath):
    with open(fpath, 'r') as f:
        json_ = json.load(f)
    return json_

def F_subdir_file_list(dir_path,matcher=lambda x:True):
    f = []
    # load all files, recursively descend into dirs
    for item in os.listdir(dir_path):
        full_path = os.path.join(dir_path,item)
        if os.path.isfile(full_path) and matcher(item):
            f.append(full_path)
        elif os.path.isdir(full_path):
            f.extend(F_subdir_file_list(os.path.join(dir_path,item),matcher))
    return f


def F_list_files(folder):
    onlyfiles = []
    for f in os.listdir(folder):
        fullpath = os.path.join(folder, f)
        if os.path.isfile(fullpath):
            onlyfiles.append(fullpath)
    return onlyfiles




