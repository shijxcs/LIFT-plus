import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Linear):
    def __init__(self, feat_dim, num_classes, bias=True, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
    
    def reset_parameters(self):
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class CosineClassifier(LinearClassifier):
    def __init__(self, feat_dim, num_classes, scale=30, bias=False, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
        self.scale = scale

    def forward(self, x):
        return F.linear(self.scale * F.normalize(x), F.normalize(self.weight), self.bias)


class L2NormClassifier(LinearClassifier):
    def __init__(self, feat_dim, num_classes, bias=False, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
    
    def forward(self, x):
        return F.linear(x, F.normalize(self.weight), self.bias)


class LayerNormClassifier(LinearClassifier):
    def __init__(self, feat_dim, num_classes, bias=False, dtype=None, device=None, **kwargs):
        super().__init__(feat_dim, num_classes, bias, dtype=dtype, device=device)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype, device=device)

    def forward(self, x):
        return F.linear(self.ln(x), F.normalize(self.weight), self.bias)
