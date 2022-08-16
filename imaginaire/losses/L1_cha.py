import torch
from torch import nn


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        numel = torch.numel(diff)
        numel = numel * numel
        error = torch.sqrt(diff * diff/numel + self.eps)
        loss = torch.sum(error)
        return loss