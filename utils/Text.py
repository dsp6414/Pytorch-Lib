# -*- coding: utf-8 -*-
"""Utilities for text input preprocessing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import random

from six import iterkeys, iteritems, u, string_types

from six.moves import range
from six.moves import zip

import torch
from smart_open import *

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def encode(input_lines, word_dict):
    """
    encode list of strings into word-level representation
    """
    lines = list(map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines


def encode2Tensor(input_lines, word_dict, unk):
    """
    encode list of strings into word-level representation (tensor) with unk
    """
    lines = list(map(lambda t: torch.LongTensor(list(map(lambda m: word_dict.get(m, unk), t))), input_lines))
    return lines

def one_hot_(y, num_classes=None):

    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments

        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """

    y = np.array(y, dtype='int')

    input_shape = y.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:

        input_shape = tuple(input_shape[:-1])

    y = y.ravel()

    if not num_classes:

        num_classes = np.max(y) + 1

    n = y.shape[0]

    categorical = np.zeros((n, num_classes), dtype=np.float32)

    categorical[np.arange(n), y] = 1

    output_shape = input_shape + (num_classes,)

    categorical = np.reshape(categorical, output_shape)

    return categorical

def any2utf8(text, errors='strict', encoding='utf8'):

    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):

    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode

def lower_to_unicode(text, encoding='utf8', errors='strict'):
    """Lowercase `text` and convert to unicode."""
    return to_unicode(text.lower(), encoding, errors)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def pad_sequences(sequences, pad, max_length,is_pad_end=True):
    """
    Args:
        sequences: a generator of list or tuple.
        pad: the value to pad with.
    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded = []

    for seq in sequences:
        seq = list(seq)
        if is_pad_end:
            seq_ = seq[:max_length] + [pad] * max(max_length - len(seq), 0)
        else:
            seq_ = [pad] * max(max_length - len(seq), 0) + seq[:max_length]
        sequence_padded += [seq_]
        

    return sequence_padded

def load_glove_vocab(filename):
    """Loads GloVe's vocab from a file.
    Args:
        filename (str): path to the glove vectors.
    Returns:
        set: a set of all words in GloVe.
    """
    print('Building vocab...')
    with open(filename,encoding='utf-8') as f:
        vocab = {line.strip().split()[0] for line in f} #set
    print('- done. {} tokens'.format(len(vocab)))
    return dict(enumerate(vocab))  #return set

def load_word_embeddings(vocab, glove_filename, dim):
    """Loads GloVe vectors in numpy array.
    Args:
        vocab (): dictionary vocab[word] = index.
        glove_filename (str): a path to a glove file.
        dim (int): dimension of embeddings.
    Returns:
        numpy array: an array of word embeddings.
    """
    embeddings = np.zeros([len(vocab)+1, dim],dtype=np.float32) #0为PAD
    with open(glove_filename,encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding,dtype=np.float32)

    return embeddings


def word2idx(sents, word2idx):
    return [[word2idx[w] for w in s if w in word2idx] for s in sents] 

def revdict(d):
    return {v: k for (k, v) in iteritems(dict(d))}

def get_entities(seq):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> print(get_entities(seq))
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    while i < len(seq):
        if seq[i].startswith('B'):
            for j in range(i+1, len(seq)):
                if seq[j].startswith('I') and types[j] == types[i]:
                    continue
                break
            chunks.append((types[i], i, j))
            i = j
        else:
            i += 1
    return chunks

def utf_8_encoder(doc):

    for line in doc:

        yield line.encode('utf-8')


def skipgrams(sequence, vocabulary_size,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None, seed=None):
    """Generates skipgram word pairs.

    Takes a sequence (list of indexes of words),
    returns couples of [word_index, other_word index] and labels (1s or 0s),
    where label = 1 if 'other_word' belongs to the context of 'word',
    and label=0 if 'other_word' is randomly sampled

    # Arguments
        sequence: a word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: int. maximum possible word index + 1
        window_size: int. actually half-window.
            The window of a word wi will be [i-window_size, i+window_size+1]
        negative_samples: float >= 0. 0 for no negative (=random) samples.
            1 for same number as positive samples. etc.
        shuffle: whether to shuffle the word couples before returning them.
        categorical: bool. if False, labels will be
            integers (eg. [0, 1, 1 .. ]),
            if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]
        sampling_table: 1D array of size `vocabulary_size` where the entry i
            encodes the probability to sample a word of rank i.
        seed: random seed.

    # Returns
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.

    # Note
        By convention, index 0 in the vocabulary is
        a non-word and will be skipped.
    """
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0, 1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [[words[i % len(words)],
                    random.randint(1, vocabulary_size - 1)] for i in range(num_negative_samples)]
        if categorical:
            labels += [[1, 0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels

def remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: int, maximum length
        seq: list of lists where each sublist is a sequence
        label: list where each element is an integer

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label

class tDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.wordfreq={}
        self.len = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.len
            self.wordfreq[word]=1
            self.len +=1
        else:
            self.wordfreq[word] = self.wordfreq[word]+1

    def __len__(self):
        return len(self.word2idx)

    def revdict(self):
        self.idx2word = {v:k for (k, v) in iteritems(self.word2idx)}

    def docs2idx(self,sents):
        return [[self.word2idx[w] for w in s if w in self.word2idx] for s in sents] 

    def build_vocab(self,sents):
        for s in sents:
            for w in s:
                self.add_word(w)
        self.revdict()





