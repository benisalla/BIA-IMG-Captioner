import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange


class BIAConv2DEmbedding(nn.Module):
    """
    This layer simplifies and enhances the processing of images by extracting patch-level embeddings,
    making the process efficient and straightforward.

    NOTE: This is not feature extraction in the traditional CNN sense; instead, it's a helper
    layer for processing the entire image into patch embeddings.

    Parameters:
    -----------
    n_emb : int
        Number of embedding dimensions.
    p_size : int
        Patch size for the convolution operation.
    c_dim : int
        Number of input channels in the image.
    """

    def __init__(self, n_emb: int, p_size: int, c_dim: int) -> None:
        super(BIAConv2DEmbedding, self).__init__()
        self.p_size = p_size
        self.ite = nn.Sequential(
            nn.Conv2d(c_dim, n_emb, kernel_size=p_size, stride=p_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_emb))

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, H, W)
        b, _, _, _ = x.shape
        x = self.ite(x)
        return x
