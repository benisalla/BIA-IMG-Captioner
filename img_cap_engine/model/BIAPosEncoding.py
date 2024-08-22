import torch
import torch.nn as nn
from torch import Tensor


class BIAPosEncoding(nn.Module):
    """
    This layer adds learnable position encodings to help the model retain the
    positional information of tokens in the sequence, differing from the fixed
    position encodings used in traditional transformer architectures.

    Parameters:
    -----------
    n_emb : int
        The number of dimensions for each embedding.
    p_size : int
        The size of each patch in the image.
    im_size : int
        The size of the input image (assumed to be square).
    """

    def __init__(self, n_emb: int, p_size: int, im_size: int) -> None:
        super(BIAPosEncoding, self).__init__()

        self.ppe = nn.Parameter(torch.randn((im_size // p_size) ** 2, n_emb))

    def forward(self, x: Tensor) -> Tensor:
        x += self.ppe
        return x
