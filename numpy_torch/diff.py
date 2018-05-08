import os
import torch
import numpy as np
from torch.autograd import Variable

x = np.random.rand(3,2,4)
print(x.shape)
y = torch.Tensor(x)
print(y.shape)
print(y.index_select(0,torch.LongTensor([0,1])))
y_var = Variable(y)
print(y_var.permute(2,1,0).shape)

x_new = x[:,np.newaxis,:,:]
print(x_new.shape)

y_new = torch.unsqueeze(y,dim=1)
print(y_new.shape)

x_new = np.transpose(x,axes=(2,0,1))
print(x_new.shape)


y_var = Variable(torch.Tensor(x))
print(y_var)

x = np.arange(10)
print(x)
y = torch.arange(10)
print(y)

inds = [2,3,5]
print(x[inds])
print(y[inds])

print(x[0:10:2])
print(y[0:10:2])

print(x[-3:-1])
print(y[-3:-1])
print(x[x < 5])
print(y[y < 5])

x = np.array([[ 0, 1, 2],
    [ 3, 4, 5],
    [ 6, 7, 8],
    [ 9, 10, 11]])
rows = (x.sum(-1) % 2) == 0

columns = [0, 2]
print(x[np.ix_(rows, columns)])

a = np.arange(6).reshape(2,3)
for x in np.nditer(a):
    print (x)

a = np.arange(3)
b = np.arange(6).reshape(2,3)
for x, y in np.nditer([a,b]):
    print ("%d:%d" % (x,y))