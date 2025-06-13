import torch
from torch import nn




class TimeDistributed(nn.Module):
    """
    This module can wrap any given module and stacks the time dimension with the batch dimension of the inputs
    before applying the module.
    Borrowed from this fruitful `discussion thread
    <https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4>`_.

    Parameters
    ----------
    module : nn.Module
        The wrapped module.
    batch_first: bool
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    return_reshaped: bool
        A boolean indicating whether to return the output in the corresponding original shape or not.
    """

    def __init__(self, module: nn.Module, batch_first: bool = True, return_reshaped: bool = True):
        super(TimeDistributed, self).__init__()
        self.module: nn.Module = module  # the wrapped module
        self.batch_first: bool = batch_first  # indicates the dimensions order of the sequential data.
        self.return_reshaped: bool = return_reshaped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If this is “static” (no time dim), treat it the same as a single time-step
        if x.dim() <= 2:
            # x: [batch, features]
            y = self.module(x)  # should return a Tensor [batch, out_features]
            return y

        # Now x is [batch, time, features]
        B = x.size(0)
        T = x.size(1)
        F_in = x.size(2)
        # flatten batch & time
        x_flat = x.contiguous().view(B * T, F_in)     # [B*T, F_in]
        y_flat = self.module(x_flat)                  # [B*T, F_out]

        if self.return_reshaped:
            F_out = y_flat.size(1)
            if self.batch_first:
                # reshape to [batch, time, out_features]
                return y_flat.contiguous().view(B, T, F_out)
            else:
                # reshape to [time, batch, out_features]
                return y_flat.contiguous().view(T, B, F_out)

        # otherwise, just return the flat output
        return y_flat


class NullTransform(nn.Module):
    def __init__(self):
        super(NullTransform, self).__init__()

    @staticmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # completely ignore x and return an “empty” embedding tensor
        # shape: [batch*time_steps, 0] so concatenation still works
        # if x is 2-D ([batch, features]) or 3-D ([batch, time, features]), flatten first
        flat = x.view(-1, x.shape[-1]) if x.dim() > 1 else x.unsqueeze(1)
        return flat.new_empty((flat.size(0), 0))
