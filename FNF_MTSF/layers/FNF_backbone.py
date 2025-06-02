__all__ = ['FNF_backbone']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.FNF_layers import *
from layers.RevIN import RevIN

from layers.Embed import PatchEmbedding
from layers.Fourier import FourierNeuralFilter, GateLayer


class FNF_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch=None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine=True, subtract_last=False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = int((context_window - patch_len)/stride + 2)

        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, norm=norm,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        self.emb = PatchEmbedding(patch_size=patch_len, stride=stride, d_model=d_model, dropout=dropout)
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                  
        
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        z = self.emb(z)                                                                    
        
        z = self.backbone(z)                                                               
        z = self.head(z)                                                                    
        
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z


class TSTiEncoder(nn.Module):  
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act='gelu', store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model) 
        self.seq_len = q_len

        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        self.dropout = nn.Dropout(dropout)



        self.layers1 = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                        attn_dropout=attn_dropout, dropout=dropout,
                                                        activation='gelu', res_attention=res_attention,
                                                        pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.layers2 = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                        attn_dropout=attn_dropout, dropout=dropout,
                                                        activation='gelu', res_attention=res_attention,
                                                        pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.gate = GateLayer(d_model)
        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_num x d_model]
        
        n_vars = x.shape[1]
        patch_num = x.shape[2]

        u1 = x
        u1 = torch.reshape(u1, (u1.shape[0]*u1.shape[1],u1.shape[2],u1.shape[3]))      # u1: [bs * nvars x patch_num x d_model]
        for mod in self.layers1: u1 = mod(u1)                                                     # u1: [bs * nvars x patch_num x d_model]
        u1 = torch.reshape(u1, (-1, n_vars,u1.shape[-2],u1.shape[-1]))                # u1: [bs x nvars x patch_num x d_model]

        u2 = x.permute(0,2,1,3)                                                     # u2: [bs x patch_num x nvars x d_model]
        u2 = torch.reshape(u2, (u2.shape[0]*u2.shape[1],u2.shape[2],u2.shape[3]))      # u2: [bs * patch_num x nvars x d_model]
        for mod in self.layers2: u2 = mod(u2)                                                     # u2: [bs * patch_num x nvars x d_model]
        u2 = torch.reshape(u2, (-1, patch_num,u2.shape[-2],u2.shape[-1]))                # u2: [bs x patch_num x nvars x d_model]
        u2 = u2.permute(0,2,1,3)                                                      # u2: [bs x nvars x patch_num x d_model]

        z = self.gate(u1, u2)

        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()

        self.self_attn = FourierNeuralFilter(d_model=d_model, num_heads=n_heads, dropout=attn_dropout, d_ff=d_ff)

        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm

    def forward(self, src:Tensor) -> Tensor:
        if self.pre_norm:
            src = self.norm_attn(src)

        src2 = self.self_attn(src)

        src = src + self.dropout_attn(src2)

        if not self.pre_norm:
            src = self.norm_attn(src)

        return src
        

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):    

        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])         
                z = self.linears[i](z)                  
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x