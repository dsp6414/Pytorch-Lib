import _init_paths

import utils
from utils.Corpus import *
from utils.Text import *

import numpy as np
import torch
import os





if __name__ == '__main__':
   
   
    #glove_filename = 'F:\\My Research\\最近研究\\Pytorch\\PythonApplication1\\PythonApplication1\\glove.6B\\glove.6B.300d.txt'
    #emb = load_word_embeddings(word_2_idx,glove_filename,dim=300) #最后一个是PAD


    _dict = tDictionary()
    docs = [['Hello'],['China'],['Hello','China']]    
    _dict.build_vocab(docs)
    docs1 = [['my','girl'],['China'],['book','prices']]
    _dict.build_vocab(docs1)
    print(_dict.word2idx)
    print(_dict.len)
    print(_dict.wordfreq)
    print(_dict.idx2word)

    idx=_dict.docs2idx(docs)
    print(idx)

    vocabs = len(_dict)

    docs = pad_sequences(idx, vocabs, 10,is_pad_end=True)
    print(docs)
    coded = one_hot_([1,4,3],6)
    print(coded)
    print('finished')
