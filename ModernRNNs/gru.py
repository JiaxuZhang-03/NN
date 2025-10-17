import torch
from torch import nn
from d2l import torch as d2l

batch_size,num_steps = 32,35
train_iter,vocab = [],[]

def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size = shape,device=device)*0.01
    def three():
        return (normal((num_inputs,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros(num_hiddens,device = device))
    
    W_xz,W_hz,b_z = three()
    W_xr,W_hr,b_r = three()
    W_xh,W_hh,b_h = three()
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device = device)
    params = [W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)

def gru(inputs,state,params):
    W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X@W_xz)+(H*W_hz)+b_z)
        R = torch.sigmoid((X@W_xr) + (H@W_hr) + b_r)
        H_tilda = torch.tanh((X@W_xh) + ((R*H)@W_hh) + b_h)
        H = Z*H + (1-Z)*H_tilda
        Y = H@W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs,dim = 0), (H,)

vocab_size,num_hiddens,device = len(vocab),256,torch.device('mps')
num_epochs,lr = 500,1
model = d2l.RNNModelScratch(len(vocab),num_hiddens,device,get_params,init_gru_state,gru)
d2l.train_ch8(model,train_iter,vocab,lr,num_epochs,device)

# OR: modern APIs:
class GRULanguageModel(nn.Module):
    def __init__(self,vocab_size,num_hiddens,device):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size,num_hiddens)
        self.gru = nn.GRU(num_hiddens,num_hiddens,batch_first=True)
        self.decoder = nn.Linear(num_hiddens,vocab_size)
        self.device = device
    def forward(self,X,state = None):
        X = self.embedding(X)
        Y,state = self.gru(X,state)
        output = self.decoder(Y)
        return output, state
    def begin_state(self,batch_size):
        return torch.zeros((1,batch_size,self.num_hiddens),device = self.device)
    

def train_epoch(model,data_iter,loss_fn,optimizer,device):
    model.train()
    total_loss,count = 0,0
    for X,Y in data_iter:
        X,Y = X.to(device),Y.to(device)
        batch_size = X.shape[0]
        state = model.begin_state(batch_size)
        Y_hat,state = model(X,state)
        loss = loss_fn(Y_hat.reshape(-1,Y_hat.shape[1]),Y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Y.numel()
        count += Y.numel()
    return total_loss/count


def train(model,data_iter,vocab_size,lr,num_epochs,device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in range(num_epochs):
        ppl = torch.exp(torch.tensor(train_epoch(model,data_iter,loss_fn,optimizer,device)))
        print(f"{epoch+1} epoch, with Perplexity : {ppl:.2f}")
        



