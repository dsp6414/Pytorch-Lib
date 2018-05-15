import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class CoattentionEncoder(nn.Module):
    
    def __init__(self, vocab_size,embedding_size,hidden_size,n_layer=1,dropout_p=0.3,use_cove=False,use_cuda=False):
        super(CoattentionEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.embedding = nn.Embedding(vocab_size,embedding_size, padding_idx=0) # shared embedding
        self.enc_lstm = nn.LSTM(embedding_size,hidden_size,n_layer,batch_first=True)
        self.coattn_lstm = nn.LSTM(hidden_size*3,hidden_size,batch_first=True,bidirectional=True)
        self.q_linear = nn.Linear(hidden_size,hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.use_cuda = use_cuda
        self.use_cove = use_cove
        
        if self.use_cove:
            self.mtlstm = nn.LSTM(300, 300, num_layers=2, bidirectional=True)
            self.mtlstm.load_state_dict(torch.load(THIS_PATH+'/models/wmtlstm-b142a7f2.pth')) # Cove
        
        if use_cuda:
            self.cuda()
        
    def init_embed(self,pretrained_wvectors,is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_wvectors).float())
        
        if is_static:
            self.embedding.weight.requires_grad = False
            
    def init_hidden(self,size,direc=1,dim=1):
        hidden = Variable(torch.zeros(direc*self.n_layer,size,self.hidden_size*dim))
        context = Variable(torch.zeros(direc*self.n_layer,size,self.hidden_size*dim))
        if self.use_cuda:
            hidden=hidden.cuda()
            context=context.cuda()
        return (hidden,context)
    
    def forward(self,documents,questions,doc_lens,question_lens,is_training=False):
        """
        documents : B,M
        questions : B,N
        """
        documents = self.embedding(documents)
        questions = self.embedding(questions)
        
        if is_training:
            documents = self.dropout(documents)
            questions = self.dropout(questions)
        
        # document encoding
        enc_hidden = self.init_hidden(documents.size(0))
        lens, indices = torch.sort(doc_lens, 0, True)
        packed_docs = pack(documents[indices], lens.tolist(), batch_first=True)
        d_o,h = self.enc_lstm(packed_docs,enc_hidden)
        d_o = unpack(d_o,batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        d_o = d_o[_indices]
        sentinel = Variable(torch.zeros(documents.size(0),1,self.hidden_size))
        if self.use_cuda:
            sentinel = sentinel.cuda()
        D = torch.cat([d_o,sentinel],1) # B,M+1,D
        
        # question encoding
        enc_hidden = self.init_hidden(questions.size(0))
        lens, indices = torch.sort(question_lens, 0, True)
        packed_questions = pack(questions[indices], lens.tolist(), batch_first=True)
        q_o,h = self.enc_lstm(packed_questions,enc_hidden)
        q_o = unpack(q_o,batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        q_o = q_o[_indices]
        sentinel = Variable(torch.zeros(questions.size(0),1,self.hidden_size))
        if self.use_cuda:
            sentinel = sentinel.cuda()
        Q_prime = torch.cat([q_o,sentinel],1)
        Q = F.tanh(self.q_linear(Q_prime.view(Q_prime.size(0)*Q_prime.size(1),-1)))
        Q = Q.view(Q_prime.size(0),Q_prime.size(1),-1)  # B,N+1,D
        
        # Affinity Matrix
        L = torch.bmm(D,Q.transpose(1,2)) # Bx(M+1)x(N+1) Affinity Matrix
        attn_D,attn_Q=[],[]
        for i in range(L.size(0)):
            attn_Q.append(F.softmax(L[i],1).unsqueeze(0)) # (M+1)x(N+1)
            attn_D.append(F.softmax(L[i].transpose(0,1),1).unsqueeze(0)) # (N+1)x(M+1)
        attn_D = torch.cat(attn_D) # Bx(N+1)x(M+1)
        attn_Q = torch.cat(attn_Q) # Bx(M+1)x(N+1)
            
        context_Q = torch.bmm(D.transpose(1,2),attn_Q).transpose(1,2) # B,N+1,D
        cat_Q = torch.cat([Q,context_Q],2) # B,(N+1),2D
        
        # context_D
        coattention = torch.bmm(cat_Q.transpose(1,2),attn_D).transpose(1,2) # B,M+1,2D
        
        coattn_hidden = self.init_hidden(coattention.size(0),2) # bidirectional
        U, _ = self.coattn_lstm(torch.cat([D,coattention],2),coattn_hidden)
        
        return  U[:,:-1] # B,M,2D
    
    
class Maxout(nn.Module):
    # https://github.com/pytorch/pytorch/issues/805
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m
    
    
class HMN(nn.Module):
    def __init__(self, hidden_size,pooling_size=8):
        super(HMN, self).__init__()
        self.hidden_size = hidden_size
        self.pooling_size = pooling_size
        self.W_D = nn.Linear(5*hidden_size,hidden_size)
        self.W_1 = Maxout(hidden_size*3,hidden_size,pooling_size)
        self.W_2 = Maxout(hidden_size,hidden_size,pooling_size)
        self.W_3 = Maxout(hidden_size*2,1,pooling_size)
        
    def forward(self,u_t,u_s,u_e,h):
        """
        u_t : Bx2D
        u_s : Bx2D
        u_e : Bx2D
        h : BxD
        """
        r = F.tanh(self.W_D(torch.cat([u_s,u_e,h],1))) # Bx5D
        m_1 = self.W_1(torch.cat([u_t,r],1))
        m_2 = self.W_2(m_1)
        m_3 = self.W_3(torch.cat([m_1,m_2],1))
        
        return m_3
    
    
class DynamicDecoder(nn.Module):
    def __init__(self,hidden_size,pooling_size=8,dropout_p=0.3,max_iter=4,use_cuda=False):
        super(DynamicDecoder,self).__init__()
        
        self.hidden_size = hidden_size
        self.max_iter = max_iter
        self.dec_lstm_cell = nn.LSTMCell(hidden_size*4,hidden_size)
        self.hmn_start = HMN(hidden_size,pooling_size)
        self.hmn_end = HMN(hidden_size,pooling_size)
        self.dropout = nn.Dropout(dropout_p)
        self.use_cuda = use_cuda
        
        if use_cuda:
            self.cuda()
        
    def init_hidden(self,size):
        hidden = Variable(torch.zeros(size,self.hidden_size))
        context = Variable(torch.zeros(size,self.hidden_size))
        if self.use_cuda:
            hidden=hidden.cuda()
            context=context.cuda()
        return (hidden,context)
        
    def forward(self,U,is_training=False):
        """
        U : B,M,2D
        """
        hidden = self.init_hidden(U.size(0))
        si,ei = 0,1 
        u_s = torch.cat([u[si].unsqueeze(0) for u in U]) # Bx2D
        u_e = torch.cat([u[ei].unsqueeze(0) for u in U]) # Bx2D
        entropies=[]
        for i in range(self.max_iter):
            entropy=[]
            alphas=[]
            for u_t in U.transpose(0,1): # M,B,2D
                alphas.append(self.hmn_start(u_t,u_s,u_e,hidden[0])) # B,M
            
            alpha = torch.cat(alphas,1)
            entropy.append(alpha)
            alpha = alpha.max(1)[1] # B
            u_s = torch.cat([U[i][alpha.data[i]].unsqueeze(0) for i in range(U.size(0))]) # Bx2D
            
            betas=[]
            for u_t in U.transpose(0,1):
                betas.append(self.hmn_end(u_t,u_s,u_e,hidden[0]))
            beta = torch.cat(betas,1)
            entropy.append(beta)
            beta = beta.max(1)[1] # B
            u_e = torch.cat([U[i][beta.data[i]].unsqueeze(0) for i in range(U.size(0))]) # Bx2D
            
            hidden = self.dec_lstm_cell(torch.cat([u_s,u_e],1),hidden) 
            
            if is_training == False and si == alpha.data[0] and ei == beta.data[0]:
                entropies.append(entropy)
                break
            else:
                entropies.append(entropy)
                si=alpha.data[0]
                ei=beta.data[0]
            
        return alpha,beta, entropies