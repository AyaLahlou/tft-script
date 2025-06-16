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
        # If no time dimension, add one so flatten() logic is uniform
        if x.dim() <= 2:
            if self.batch_first:
                x = x.unsqueeze(1)  # (batch, 1, features)
            else:
                x = x.unsqueeze(0)  # (1, time, features)

        # Flatten batch & time dims into one for the wrapped module
        if self.batch_first:
            # (batch, time, features) -> (batch*time, features)
            x_reshape = x.flatten(0, 1)
        else:
            # (time, batch, features) -> (batch*time, features)
            x_reshape = x.flatten(1, 2).transpose(0, 1)

        # Apply the wrapped module
        raw = self.module(x_reshape)

        # If it returned a list of tensors, stack into one Tensor
        if isinstance(raw, list):
            y = torch.stack(raw, dim=0)
        else:
            y = raw

        # Reshape back to (batch, time, features)
        if self.batch_first:
            # y: (batch*time, features) -> (batch, time, features)
            return y.view(x.size(0), -1, y.size(-1))
        else:
            # y: (batch*time, features) -> (time, batch, features)
            return y.view(-1, x.size(1), y.size(-1)).transpose(0, 1)


class NullTransform(nn.Module):
    def __init__(self):
        super(NullTransform, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty(0, device=x.device)
