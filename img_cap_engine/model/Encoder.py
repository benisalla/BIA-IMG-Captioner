import math
from img_cap_engine.model.BIAConv2DEmbedding import BIAConv2DEmbedding
from img_cap_engine.model.BIAPosEncoding import BIAPosEncoding
from img_cap_engine.model.ResFFNet import ResFFNet
from img_cap_engine.model.ResMHAtten import ResMHAtten
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class Encoder(nn.Module):
    """
    An encoder module designed for processing images. This layer is built to extract
    and learn the features of images, ensuring an efficient image captioning process.

    Parameters:
    -----------
    n_block : int
        The number of transformer blocks in the encoder.
    n_emb : int
        The number of embedding dimensions.
    n_head : int
        The number of attention heads.
    h_size : int
        The dimension of each attention head.
    p_size : int
        The patch size used to divide the image for convolutional embedding.
    im_size : int
        The size of the input image (assumed to be square).
    c_dim : int
        The number of input channels in the image.
    exp_fac : int, optional
        Expansion factor for the feed-forward network hidden layer, by default 4.
    d_rate : float, optional
        Dropout rate applied to various layers, by default 0.0.
    bias : bool, optional
        Whether to include bias terms in the layers, by default False.
    device : str, optional
        The device on which to run the computations ('cpu' or 'cuda'), by default 'cpu'.

    Attributes:
    -----------
    pte : nn.Module
        The convolutional embedding layer that processes the image into patch embeddings.
    ppe : nn.Module
        The positional encoding layer that adds learnable position encodings to the patch embeddings.
    drop : nn.Dropout
        Dropout layer applied to embeddings.
    blocks : nn.ModuleList
        A list of transformer blocks, each containing a multi-head self-attention layer and a feed-forward network.

    """

    def __init__(
        self,
        n_block: int,
        n_emb: int,
        n_head: int,
        h_size: int,
        p_size: int,
        im_size: int,
        c_dim: int,
        exp_fac: int = 4,
        d_rate: float = 0.0,
        bias: bool = False,
        device: str = "cpu",
    ) -> None:
        super(Encoder, self).__init__()
        self.device = device

        assert (
            n_block is not None
            and p_size is not None
            and n_emb is not None
            and h_size is not None
        ), "Configuration must include n_block, p_size, embd_dim, and head_dim."

        self.n_block = n_block
        self.n_patch = im_size**2 // p_size**2
        self.n_emb = n_emb
        self.h_size = h_size
        self.p_size = p_size
        self.im_size = im_size
        self.c_dim = c_dim
        self.d_rate = d_rate
        self.bias = bias
        self.n_head = n_head

        self.pte = BIAConv2DEmbedding(
            n_emb=self.n_emb, p_size=self.p_size, c_dim=self.c_dim
        )
        self.ppe = BIAPosEncoding(
            n_emb=self.n_emb, p_size=self.p_size, im_size=self.im_size
        )
        self.drop = nn.Dropout(self.d_rate)

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResMHAtten(
                            n_emb=n_emb,
                            n_head=n_head,
                            h_size=h_size,
                            d_rate=d_rate,
                            is_decoder=False,
                            device=device,
                        ),
                        ResFFNet(n_emb=n_emb, exp_fac=exp_fac, d_rate=d_rate),
                    ]
                )
                for _ in range(n_block)
            ]
        )

        self.apply(self._init_weights)

        for par_name, par in self.named_parameters():
            if par_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    par, mean=0.0, std=0.02 / math.sqrt(2 * self.n_block)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, BIAConv2DEmbedding):
            torch.nn.init.normal_(module.ite[0].weight, mean=0.0, std=std)
            if module.ite[0].bias is not None:
                torch.nn.init.zeros_(module.ite[0].bias)
            torch.nn.init.normal_(module.cls_token, mean=0.0, std=std)
        elif isinstance(module, BIAPosEncoding):
            torch.nn.init.normal_(module.ppe, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_init_args(self) -> Dict[str, int]:
        return {
            "n_block": self.n_block,
            "n_emb": self.n_emb,
            "h_size": self.h_size,
            "p_size": self.p_size,
            "im_size": self.im_size,
            "c_dim": self.c_dim,
            "d_rate": self.d_rate,
            "bias": self.bias,
        }

    def forward(self, img: Tensor) -> Tensor:
        """
        Forward pass through the encoder.

        Parameters:
        -----------
        img : Tensor
            Input image tensor of shape (batch_size, channels, height, width).

        Returns:
        --------
        Tensor
            Output tensor after passing through the convolutional embedding,
            positional encoding, and transformer blocks.
        """

        x = self.pte(img).to(self.device)
        x = self.ppe(x).to(self.device)

        for attn, ffnet in self.blocks:
            x = attn(q=x, kv=x)
            x = ffnet(x)

        return x
