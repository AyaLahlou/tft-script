import torch
from torch import nn


class TimeDistributed(nn.Module):
    """
    This module wraps a given module to handle temporal data by stacking time with batch dimension.
    Simplified for TorchScript compatibility.
    """

    def __init__(self, module: nn.Module, batch_first: bool = True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 2:
            return self.module(x)

        # Squash samples and timesteps
        x_reshape = x.flatten(0, 1) if self.batch_first else x.flatten(1, 2).transpose(0, 1)
        y = self.module(x_reshape)

        # Reshape back to original dimensions
        if self.batch_first:
            y = y.view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1)).transpose(0, 1)
        return y


class NullTransform(nn.Module):
    def __init__(self):
        super(NullTransform, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty(0, device=x.device)
