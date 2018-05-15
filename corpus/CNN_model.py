import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.lookup_table = nn.Embedding(self.vocab_size+1, self.embed_dim)
        self.lookup_table.weight.data.copy_(torch.from_numpy(self.emb))        
        self.lookup_table.requires_grad=False  #固定参数
        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                            nn.Conv2d(in_channels=1,
                            out_channels=self.kernel_num,
                            kernel_size=(filter_size, self.embed_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))

        self.logistic = nn.Linear(len(self.filter_sizes) * self.kernel_num,
                                self.label_size)

        self.dropout = nn.Dropout(self.dropout)

        self._init_weight()

    def forward(self, x):
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3

        x = self.lookup_table(x)
        x = x.unsqueeze(c_idx)

        enc_outs = []
        for encoder in self.encoders:
            enc_ = F.relu(encoder(x))
            k_h = enc_.size()[h_idx]
            enc_ = F.max_pool2d(enc_, kernel_size=(k_h, 1))
            enc_ = enc_.squeeze(w_idx)
            enc_ = enc_.squeeze(h_idx)
            enc_outs.append(enc_)

        encoding = self.dropout(torch.cat(enc_outs, 1))
        return F.log_softmax(self.logistic(encoding))
        #return self.logistic(encoding)

    def _init_weight(self, scope=.1):
        #self.lookup_table.weight.data.uniform_(-scope, scope) #lookup_table参数采用预训练参数
        self.logistic.weight.data.uniform_(-scope, scope)

