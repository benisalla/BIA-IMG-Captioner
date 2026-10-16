from img_cap_engine.model.BIALayerNorm import BIALayerNorm
import torch
from torch import nn


class ResFFNet(nn.Module):
    """
    Residual Feedforward Network (ResFFNet) module designed to improve learning stability and performance in transformer models.
    It applies a feedforward neural network with a residual connection, which helps in preserving the original input while 
    adding non-linear transformations.

    Parameters:
    -----------
    n_emb : int
        The dimensionality of the input embeddings.
    exp_fac : int, optional
        The expansion factor for the hidden layer in the feedforward network. This determines the size of the hidden layer 
        as `exp_fac * n_emb`. Default is 4.
    d_rate : float, optional
        The dropout rate applied to the intermediate layers and the output. Default is 0.0.
    
    Attributes:
    -----------
    ln : BIALayerNorm
        Layer normalization applied to the input tensor before passing through the feedforward layers.
    fc1 : nn.Linear
        The first linear layer that expands the input embedding dimension to `exp_fac * n_emb`.
    relu : nn.ReLU
        ReLU activation function applied after the first linear layer.
    fc2 : nn.Linear
        The second linear layer that reduces the expanded dimension back to `n_emb`.
    drop : nn.Dropout
        Dropout layer applied after each linear transformation to prevent overfitting.
    """

    def __init__(self, n_emb: int, exp_fac: int = 4, d_rate: float = 0.0) -> None:
        super(ResFFNet, self).__init__()
        self.n_emb = n_emb
        self.exp_fac = exp_fac
        self.d_rate = d_rate

        self.ln = BIALayerNorm(self.n_emb)
        self.fc1 = nn.Linear(self.n_emb, exp_fac * self.n_emb)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(exp_fac * self.n_emb, self.n_emb)
        self.drop = nn.Dropout(d_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o_x = x.clone()  # fork
        x = self.ln(x)  # norm

        # feedforward
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)

        # drop and res conn
        x = self.drop(x) + o_x
        return x
