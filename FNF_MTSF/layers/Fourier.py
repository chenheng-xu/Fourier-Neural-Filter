import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalConvolution(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        num_heads: int = 8, 
        sparsity_threshold: float = 0.01, 
        hard_thresholding_fraction: float = 1, 
        hidden_size_factor: int = 1
    ):
        super().__init__()

        self.d_model = d_model
        self.sparsity_threshold = sparsity_threshold
        self.num_heads = num_heads
        self.block_size = self.d_model // self.num_heads
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_heads, self.block_size))

    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_heads, self.block_size)
        
        o1_real = torch.zeros([B, N // 2 + 1, self.num_heads, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_heads, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :kept_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)
        
        return x


class FourierNeuralFilter(nn.Module):
    def __init__(
        self, 
        d_model: int = 128, 
        d_ff: int = 256,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

        self.conv = GlobalConvolution(d_model=d_ff//2, 
                                      num_heads=num_heads)
        
        self.out_proj = nn.Linear(d_ff//2, d_model)
        
    def forward(self, x):
        x = self.in_proj(x)
        y, z = x.chunk(2, dim=-1)  
        y = self.conv(y)
        y = self.dropout(y)
        
        x = y * F.gelu(z)
        x = self.out_proj(x)

        return x


class GateLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 1)        
    def forward(self, x1, x2):
        gate = self.gate(x1).sigmoid()
        return gate * x1 + (1 - gate) * x2