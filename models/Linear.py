import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.VariateEmbedding import VarEmbed

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len

        if self.individual:
            self.proj = nn.ModuleList()
            for i in range(self.channels):
                self.proj.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            if configs.VE_len == 0:
                self.proj = nn.Linear(self.seq_len, self.pred_len)
            else:
                self.proj = VarEmbed(configs.enc_in, configs.VE_len, configs.seq_len, configs.pred_len, configs.ParamRatio, 1, 0, configs.usePreprocess) 
                

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, V = x_enc.shape
        # RevIN
        x = x_enc   # (B, L, V)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)   # (B, L, V)

        if self.individual:
            xy_ = torch.zeros([B, self.pred_len, V],dtype=x.dtype).to(x.device)  # (B, dout, V)
            for i in range(self.channels):
                xy_[:,:,i]=self.proj[i](x[:,:,i].permute(0,1)).permute(0,1) # (B, dout, V)
        else:
            xy_ = self.proj(x.permute(0,2,1)).permute(0,2,1)    # (B, dout, V)

        x = xy_ * torch.sqrt(x_var) +x_mean
        return x