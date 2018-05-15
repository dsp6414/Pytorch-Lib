import _init_paths

import argparse
import time


import torch
from torch.autograd import Variable
from torch.optim import Adam,SGD
import torch.nn as nn
import utils
from utils.Text import load_word_embeddings


parser = argparse.ArgumentParser(description='CNN text classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training')
parser.add_argument('--save', type=str, default='./data/cnn-text-classfication/CNN_Text.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/cnn-text-classfication/corpus.pt',
                    help='location of the data corpus')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=300,
                    help='number of embedding dimension [default: 128]')
parser.add_argument('--kernel-num', type=int, default=128,
                    help='number of each kind of kernel')
parser.add_argument('--filter-sizes', type=str, default='3,4,5',
                    help='filter sizes')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', type=bool,default=True,
                    help='enables cuda')

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
################################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.vocab_size = data['dict']['vocab_size']
args.label_size = data['dict']['label_size']
args.filter_sizes = list(map(int, args.filter_sizes.split(",")))

training_data = DataLoader(
             data['train']['src'],
             data['train']['label'],
             args.max_len,
             batch_size=args.batch_size,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['label'],
              args.max_len,
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda,
              evaluation=True)

emb = load_word_embeddings(data['dict']['train'],'F:\\My Research\\最近研究\\Pytorch\\PythonApplication1\\PythonApplication1\\glove.6B\\glove.6B.300d.txt',300)
args.emb = emb
# ##############################################################################
# Build model
# ##############################################################################
from CNN_model import CNN_Text  #注意这个导入

cnn = CNN_Text(args)
if use_cuda:
    cnn = cnn.cuda()
#from ASGD import *
#encoders_params = {'params':cnn.encoders.parameters(),'lr':0.001} 
#logistic_params = {'params':cnn.logistic.parameters(),'lr':0.001} 
#params = list()
#params.append(encoders_params)
#params.append(logistic_params)

#optimizer = torch.optim.Adam(params, lr=args.lr)
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=args.lr)  #过滤固定的层
optimizer = torch.optim.Adam([p  for p in cnn.parameters() if p.requires_grad], lr=args.lr) 
#optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr)
#optimizer = AccSGD(cnn.parameters(), lr=args.lr)

criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
## ##############################################################################
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
accuracy = []

def evaluate():
    cnn.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    for data, label in tqdm(validation_data, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        data,label=data.long(),label.long()
        pred = cnn(data)
        loss = criterion(pred, label)

        eval_loss += loss.data[0]
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    return eval_loss/_size, corrects, corrects/_size * 100.0, _size

def train():
    cnn.train()
    total_loss = 0
    for data, label in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=False):
        data,label=data.long(),label.long()
        optimizer.zero_grad()

        target = cnn(data)
        loss = criterion(target, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
    return total_loss[0]/training_data.sents_size

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss*1000.)
        accuracy.append(acc / 100.)

        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 90)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = cnn.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
                "src_dict": data['dict']['train']
            }
            #torch.save(model_source, args.save)


except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
