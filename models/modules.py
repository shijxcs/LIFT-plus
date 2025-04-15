import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonHalfParameter(nn.Parameter):
    def half(self):
        return self


class NonHalfLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None):
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype=torch.float32)
        if self.weight is not None:
            self.weight = NonHalfParameter(self.weight)
        if self.bias is not None:
            self.bias = NonHalfParameter(self.bias)
    
    def forward(self, input: torch.Tensor):
        return super().forward(input.type(torch.float32)).type(input.dtype)


class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, out_dim=None, dtype=None, device=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim

        self.down = nn.Parameter(torch.zeros(in_dim, bottle_dim, dtype=dtype, device=device))
        self.up = nn.Parameter(torch.zeros(bottle_dim, out_dim, dtype=dtype, device=device))
        self.scale = 1.0 / bottle_dim
        
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        nn.init.zeros_(self.up)

    def forward(self, x):
        x = x @ self.down
        x = x @ self.up
        x = x * self.scale
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, out_dim=None, dtype=None, device=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        
        self.norm = NonHalfLayerNorm(in_dim, device=device)
        self.down = nn.Linear(in_dim, bottle_dim, dtype=dtype, device=device)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottle_dim, out_dim, dtype=dtype, device=device)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x


class AdaptFormer(Adapter):
    def __init__(self, in_dim, bottle_dim, out_dim=None, dtype=None, device=None):
        super().__init__(in_dim, bottle_dim, out_dim=out_dim, dtype=dtype, device=device)
        self.scale = nn.Parameter(torch.ones([], dtype=dtype, device=device))

    def forward(self, x):
        x = super().forward(x)
        x = x * self.scale
        return x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None, device=None):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(in_dim, dtype=dtype, device=device))
        self.shift = nn.Parameter(torch.empty(in_dim, dtype=dtype, device=device))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    def forward(self, x, dim=-1):
        x = x.transpose(dim, -1)
        x = x * self.scale + self.shift
        x = x.transpose(dim, -1)
        return x

class MaskedParameter(nn.Module):
    def __init__(self, input, mask):
        super().__init__()
        self.optimized_params = nn.Parameter(torch.masked_select(input, mask=mask).detach())
        self.register_buffer("mask", mask, persistent=True)  # saved in state_dict
        self.register_buffer("input", input, persistent=False)  # not saved in state_dict
    
    def forward(self):
        return torch.masked_scatter(self.input, mask=self.mask, source=self.optimized_params)


class MaskedLinear(nn.Module):
    def __init__(self, weight, bias, weight_mask, bias_mask):
        super().__init__()
        self.masked_weight = MaskedParameter(weight, weight_mask)
        self.masked_bias = MaskedParameter(bias, bias_mask)
    
    def forward(self, x):
        return F.linear(x, self.masked_weight(), self.masked_bias())
