import torch

import argparse
import os

from const import *
from gensim.corpora.dictionary import Dictionary
from smart_open import smart_open
from utils import to_unicode

def word2idx(sents, word2idx):
    return [[word2idx[w] for w in s if w in word2idx] for s in sents] 

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class Dict(object):
    def __init__(self, word2idx={}, idx_num=0):
        self.word2idx = word2idx
        self.idx = idx_num

    def _add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def _convert(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))


class Labels(Dict):
    def __init__(self):
        super().__init__()

    def __call__(self, labels):
        _labels = set(labels)
        for label in _labels:
            self._add(label)

class Corpus(object):
    def __init__(self, path, save_data, max_len=16):
        self.train = os.path.join(path, "train")
        self.valid = os.path.join(path, "valid")
        self._save_data = save_data
        self.train_sents=[]
        self.train_labels=[]
        self.valid_sents=[]
        self.valid_labels=[]
        self.max_len = max_len
        self.dict = Dictionary()
        self.l = Labels()
           

    def parse_data_from_file(self, _file,is_train=True):
       
        _sents, _labels = [], []
        for sentence in open(_file):
            label, _, _words = sentence.replace('\xf0', ' ').partition(' ') #特定格式：类别 文档，可改写该段代码
            label = label.split(":")[0]

            words = _words.lower().strip().split()

            if len(words) > self.max_len:
                words = words[:self.max_len]

            _sents += [words]
            _labels += [label]
        if is_train:
            self.train_sents.extend(_sents)
            self.train_labels.extend(_labels)
            self.l(self.train_labels)
            self.build_dict(self.train_sents)
        else:
            self.valid_sents.extend(_sents)
            self.valid_labels.extend(_labels)

    def parse_data_from_dir(self,dirs,is_train=True,lines_are_documents=True):
        _sents, _labels = [], []
        dirs = os.path.expanduser(dirs)
        for label in sorted(os.listdir(dirs)):
            d = os.path.join(dirs, label)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    with open(path, 'rt') as f:
                        if lines_are_documents:
                            for line in f:
                                _sents += [line.lower().strip().split()]  
                                _labels += [label]                              
                        else:
                            _sents +=[f.read().strip().split()] 
                            _labels += [label]
        if is_train:
            self.train_sents.extend(_sents)
            self.train_labels.extend(_labels)
            self.l(self.train_labels)
            self.build_dict(self.train_sents)
        else:
            self.valid_sents.extend(_sents)
            self.valid_labels.extend(_labels)               


    def build_dict(self,_sents):
        self.dict.add_documents(_sents)     
    
    def build_vocab(self):
        for key in self.dict.token2id.keys():
            self.dict.token2id[key]+=1         
       

    def save(self):
        self.parse_data_from_file(self.train,is_train=True)
        self.parse_data_from_file(self.valid, is_train=False)
        #self.parse_data_from_dir("./data/corpus/data",lines_are_documents=False)
        self.build_vocab()
        data = {
            'max_len': self.max_len,
            'dict': {
                'train': self.dict.token2id,
                'vocab_size': len(self.dict),
                'label': self.l.word2idx,
                'label_size': len(self.l),
            },
            'train': {
                'doc': word2idx(self.train_sents, self.dict.token2id),
                'label': [self.l.word2idx[l] for l in self.train_labels]
            },
            'valid': {
                'doc': word2idx(self.valid_sents, self.dict.token2id),
                'label': [self.l.word2idx[l] for l in self.valid_labels]
            }
        }

        torch.save(data, self._save_data)
        print('Finish dumping the data to file - [{}]'.format(self._save_data))
        print('words length - [{}]'.format(len(self.dict)))
        print('label size - [{}]'.format(len(self.l)))
        print('train_src length - [{}]'.format(len(data['train']['doc'])))
        print('valid_src length - [{}]'.format(len(data['valid']['doc'])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='corpus processing')
    parser.add_argument('--file-path', type=str, default="./data/corpus/data",
                        help='file path')
    parser.add_argument('--save-data', type=str, default="./data/corpus/corpus.pt",
                        help='path to save processed data')
    parser.add_argument('--max-lenth', type=int, default=16,
                        help='max length left of sentence [default: 16]')
    args = parser.parse_args()
    corpus = Corpus(args.file_path, args.save_data, args.max_lenth)
    corpus.save()

  

