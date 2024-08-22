import math
from img_cap_engine.model.BIALayerNorm import BIALayerNorm
from img_cap_engine.model.ResFFNet import ResFFNet
from img_cap_engine.model.ResMHAtten import ResMHAtten
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


class Decoder(nn.Module):
    """
    The Decoder module is responsible for generating sequences, such as captions, from encoded image features.
    It leverages transformer-based architecture, including multi-head self-attention, cross-attention, and feedforward layers,
    to compute a probability distribution over a vocabulary for each position in the output sequence.

    Parameters:
    -----------
    v_size (int):
        The size of the vocabulary from which the decoder generates output sequences.
    n_emb (int):
        The dimensionality of the word embeddings.
    n_head (int):
        The number of attention heads in the multi-head attention mechanism.
    h_size (int):
        The size of the hidden layers in the attention mechanism.
    max_seq_len (int, optional):
        The maximum sequence length that the decoder can handle. Default is 10000.
    n_block (int, optional):
        The number of transformer blocks in the decoder. Default is 1.
    exp_fac (int, optional):
        The expansion factor for the hidden layers in the feedforward network. Default is 4.
    d_rate (float, optional):
        The dropout rate applied to various layers. Default is 0.0.
    device (str, optional):
        The device on which the decoder will run (e.g., 'cpu', 'cuda'). Default is "cpu".

    Attributes:
    -----------
    pte : nn.Embedding
        Embedding layer for the input tokens.
    ppe : nn.Embedding
        Positional embedding layer, which encodes the position of each token in the sequence.
    drop : nn.Dropout
        Dropout layer applied to the embeddings.
    blocks : nn.ModuleList
        A list of transformer blocks, where each block consists of a self-attention layer, a cross-attention layer,
        and a feedforward network.
    ln : BIALayerNorm
        Layer normalization applied after the transformer blocks.
    fc : nn.Linear
        Final linear layer that maps the output embeddings to the vocabulary size, producing logits for each token.
    """

    def __init__(
        self,
        v_size: int,
        n_emb: int,
        n_head: int,
        h_size: int,
        max_seq_len: int = 10000,
        n_block: int = 1,
        exp_fac: int = 4,
        d_rate: float = 0.0,
        device: str = "cpu",
    ):
        super(Decoder, self).__init__()

        self.v_size = v_size
        self.n_emb = n_emb
        self.n_head = n_head
        self.h_size = h_size
        self.n_block = n_block
        self.d_rate = d_rate
        self.exp_fac = exp_fac
        self.device = device
        self.max_seq_len = max_seq_len

        self.pte = nn.Embedding(v_size, n_emb)
        self.ppe = nn.Embedding(max_seq_len, n_emb)

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResMHAtten(
                            n_emb=n_emb,
                            n_head=n_head,
                            h_size=h_size,
                            d_rate=d_rate,
                            is_decoder=True,
                            device=device,
                        ),
                        ResMHAtten(
                            n_emb=n_emb,
                            n_head=n_head,
                            h_size=h_size,
                            d_rate=d_rate,
                            is_decoder=True,
                            device=device,
                        ),
                        ResFFNet(n_emb=n_emb, exp_fac=exp_fac, d_rate=d_rate),
                    ]
                )
                for _ in range(n_block)
            ]
        )

        self.drop = nn.Dropout(d_rate)
        self.ln = BIALayerNorm(self.n_emb)
        self.fc = nn.Linear(n_emb, v_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, BIALayerNorm):
            torch.nn.init.ones_(module.w)
            if module.b is not None:
                torch.nn.init.zeros_(module.b)

    def forward(self, idx: torch.Tensor, ex: torch.Tensor) -> torch.Tensor:
        _, dT = idx.shape

        if dT > self.max_seq_len:
            raise ValueError(
                f"Sequence length dT = {dT} exceeds maximum allowed seq_len = {self.max_seq_len}"
            )

        pos = torch.arange(dT, dtype=torch.long, device=self.device)
        pte = self.pte(idx) * math.sqrt(self.n_emb)
        ppe = self.ppe(pos)

        x = self.drop(pte + ppe)

        for self_attn, cross_attn, ffnet in self.blocks:
            x = self_attn(q=x, kv=x)
            x = cross_attn(q=x, kv=ex)
            x = ffnet(x)

        x = self.ln(x)
        logs = self.fc(x)
        return logs
