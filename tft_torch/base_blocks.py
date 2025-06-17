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
                x = x.unsqueeze(1)  # now (batch, 1, feat)
            else:
                x = x.unsqueeze(0)  # now (1, time, feat)

        # Record batch & time for the un-squeeze step
        if self.batch_first:
            batch, timesteps = x.size(0), x.size(1)
        else:
            timesteps, batch = x.size(0), x.size(1)

        # Flatten batch & time dims into one for the wrapped module
        if self.batch_first:
            x_reshape = x.flatten(0, 1)                     # (batch*time, feat)
        else:
            x_reshape = x.flatten(1, 2).transpose(0, 1)     # (batch*time, feat)

        # Apply the wrapped module
        raw = self.module(x_reshape)

        # If it returned a list of tensors, stack into one Tensor
        if isinstance(raw, list):
            # concatenate each variable’s output along the feature axis
            # each element has shape (batch*time, feat)
            y = torch.cat(raw, dim=-1)
        else:
            y = raw

        # Reshape back explicitly: (batch, time, features)
        feat_dim = y.size(-1)
        if feat_dim == 0:
            # no features: return a [batch x timesteps x 0] zero‐shape tensor
            shape = (batch, timesteps, 0)
            return torch.zeros(shape, dtype=y.dtype, device=y.device)

        if self.batch_first:
            return y.view(batch, timesteps, feat_dim)
        else:
            return y.view(batch, timesteps, feat_dim).transpose(0, 1)


class NullTransform(nn.Module):
    def __init__(self):
        super(NullTransform, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x #torch.empty(0, device=x.device)
