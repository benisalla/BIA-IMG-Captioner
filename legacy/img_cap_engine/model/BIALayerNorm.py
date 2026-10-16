import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BIALayerNorm(nn.Module):
    """
    This module normalizes the input across the features dimension, ensuring
    that the output has a mean of zero and a standard deviation of one. It's
    designed with flexibility in mind, allowing for an optional learnable bias term.

    Parameters:
    -----------
    n_dim : int
        The dimensionality of the input features to be normalized.
    bias : bool, optional
        If True, adds a learnable bias parameter to the normalized output
        (default is False).
    """

    def __init__(self, n_dim: int, bias: bool = False) -> None:
        super(BIALayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(n_dim))
        self.b = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.w.shape, self.w, self.b, 1e-5)
