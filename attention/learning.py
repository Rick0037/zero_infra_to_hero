from time import sleep

import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module): # 继承来自nn module
    # n_hidden_enc hidden layer encoder
    # n_hidden_dec hidden layer decoder
    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()
        
        self.h_hidden_dec = n_hidden_dec
        self.h_hidden_enc = n_hidden_enc
        
        self.W = nn.Linear(2*n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False) 
        self.V = nn.Parameter(torch.rand(n_hidden_dec))
        
    
    def forward(self, hidden_dec, last_layer_enc):
        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        hidden_dec = hidden_dec[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)       

        tanh_W_s_h = torch.tanh(self.W(torch.cat((hidden_dec, last_layer_enc), dim=2))) 
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)       #[b, n_hidde_dec, seq_len]
        
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  #[b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)        #[b, seq_len]
        
        att_weights = F.softmax(e, dim=1)              #[b, src_seq_len]
        
        return att_weights
    








