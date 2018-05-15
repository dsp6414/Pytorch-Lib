import argparse
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import nlp


parser = argparse.ArgumentParser(description='LSTM text classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training [default: 16]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', type=bool, default=True,help='enables cuda')

parser.add_argument('--save', type=str, default='./data/cropus/LSTM_Text.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/corpus/corpus.pt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=300,
                    help='number of embedding dimension [default: 64]')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='number of lstm hidden dimension [default: 128]')
parser.add_argument('--lstm-layers', type=int, default=1,
                    help='biLSTM layer numbers')
parser.add_argument('--bidirectional', type=int, default=True,
                    help='If True, becomes a bidirectional LSTM [default: False]')

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
###############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.vocab_size = data['dict']['vocab_size']
args.label_size = data['dict']['label_size']

training_data = DataLoader(
             data['train']['doc'],
             data['train']['label'],
             args.max_len,
             batch_size=args.batch_size,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['doc'],
              data['valid']['label'],
              args.max_len,
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda)

emb = nlp.load_pre_w2c('F:\\My Research\\最近研究\\Pytorch\\PythonApplication1\\PythonApplication1\\glove.6B\\glove.6B.300d.txt',data['dict']['train'])
args.emb = emb

# ##############################################################################
# Build model
# ##############################################################################
import model

rnn = model.LSTM_Text(args)
if use_cuda:
    rnn = rnn.cuda()


lstm_params = {'params':rnn.lstm.parameters(),'lr':0.001} 
logistic_params = {'params':rnn.logistic.parameters(),'lr':0.001} 
ln_params = {'params':rnn.ln.parameters(),'lr':0.001} 
params = list()
params.append(lstm_params)
params.append(ln_params)
params.append(logistic_params)
from ASGD import *
#optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.001)
optimizer = torch.optim.Adam(params, lr=args.lr)
#optimizer = torch.optim.SGD(params, lr=args.lr)
#optimizer = torch.optim.ASGD(params, lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

## ##############################################################################
## Training
## ##############################################################################
import time

train_loss = []
valid_loss = []
accuracy = []

def repackage_hidden(h):
    if type(h) == Variable:
        if use_cuda:
            return Variable(h.data).cuda()
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate():
    rnn.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    hidden = rnn.init_hidden()
    for data, label in validation_data:
        data,label=data.long(),label.long()
        if use_cuda:
            data,label=data.cuda(),label.cuda()
        hidden = repackage_hidden(hidden)
        pred, hidden = rnn(data, hidden)
        loss = criterion(pred, label)

        eval_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    return eval_loss[0]/_size, corrects, corrects/_size * 100.0, _size

def train():
    rnn.train()
    total_loss = 0
    hidden = rnn.init_hidden()
    for data, label in training_data:
        data,label=data.long(),label.long()  #超级坑
        if use_cuda:
            data,label=data.cuda(),label.cuda()
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        target, hidden = rnn(data, hidden)
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
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss*1000.)
        accuracy.append(acc)

        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 90)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = rnn.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
                "src_dict": data['dict']['train']
            }
            #torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
