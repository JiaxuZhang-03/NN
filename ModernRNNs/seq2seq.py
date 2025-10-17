import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqEncoder,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)
    def forward(self,X,*args):
        X = self.embedding(X)
        X = X.permute(1,0,2) #(seq_len,batch_size,embed_size)
        output,state = self.rnn(X)
        return output,state


encoder = Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
encoder.eval()
X = torch.zeros((4,7),dtype = torch.long)
output,state = encoder(X)
print(output.shape)

class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqDecoder,self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout = dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size)
    def init_state(self,enc_outputs,*args):
        return enc_outputs[1]
    def forward(self,X,state):
        X = self.embedding(X).permute(1,0,2)
        context = state[-1].repeat(X.shape[0],1,1) #(num_steps,batch_size,num_hiddens)
        X_and_context = torch.cat((X,context),2)
        output,state = self.rnn(X_and_context,state)
        output = self.dense(output).permute(1,0,2) # (batch_size,num_steps,vocab_size)
        return output,state

decoder = Seq2SeqDecoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output,state = decoder(X,state)
print(output.shape)
print(state.shape)

def sequence_mask(X,valid_len,value = 0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen),dtype = torch.float32,device = X.device)[None,:] < valid_len[:,None]
    X[~mask] = value
    return X

# X = torch.tensor([[1,2,3],[4,5,6]])
# print(sequence_mask(X,torch.tensor([1,2])))

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,pred,label,valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights,valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,self).forward(pred.permute(0,2,1),label)
        weighted_loss = (unweighted_loss * weights).mean(dim = 1)
        return weighted_loss
loss = MaskedSoftmaxCELoss()
print(loss(torch.ones(3,4,18),torch.ones((3,4),dtype = torch.long),torch.tensor([4,2,0])))





    
