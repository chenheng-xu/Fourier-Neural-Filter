import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model: int = 256, 
        max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        
        w = torch.zeros(max_len, d_model).float()
        w.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.pe = w.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        return self.pe[:, :, :x.size(2)]
    


class LearnablePositionalEmbedding(nn.Module):
    def __init__(
        self, 
        d_model: int = 256, 
        max_len: int = 5000
    ):
        super(LearnablePositionalEmbedding, self).__init__()

        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)
                
        w = torch.zeros(max_len, d_model).float()
        w.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        w = w.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(w.float())
        del w

    def forward(self, x):
        return self.pe[:, :, :x.size(2)]


class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        patch_size: int = 16, 
        stride: int = 8, 
        d_model: int = 256, 
        dropout: float = 0.1
    ):
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(patch_size, d_model, bias=False)
        self.pe = LearnablePositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        device = x.device
        B, M, _ = x.shape

        x = F.pad(x, pad=(0, self.patch_size-self.stride), mode='replicate') 
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = self.proj(x)

        _, _, N, D = x.shape
        x = x + self.pe(x).to(device)

        x = self.dropout(x)
        
        return x