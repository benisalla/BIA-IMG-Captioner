import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResMHAtten(nn.Module):
    """
    Residual Multi-Head Attention (ResMHAtten) module designed for both self-attention and cross-attention in transformer models. 
    This module applies multi-head attention with a residual connection and optional masking for autoregressive tasks in decoders.

    Parameters:
    -----------
    n_emb (int): 
        The dimensionality of the input embeddings.
    n_head (int): 
        The number of attention heads in the multi-head attention mechanism.
    h_size (int): 
        The size of the hidden layers within each attention head.
    d_rate (float): 
        The dropout rate applied to the attention scores and final output.
    is_decoder (bool, optional): 
        If True, apply a causal mask to prevent attending to future tokens. This is useful in decoder layers for autoregressive tasks. Default is False.
    device (str, optional): 
        The device on which the computation will be executed (e.g., 'cpu', 'cuda'). Default is 'cpu'.
    
    Attributes:
    -----------
    q_proj : nn.Linear
        Linear layer for projecting the query tensor.
    kv_proj : nn.Linear
        Linear layer for projecting the key and value tensors.
    o_proj : nn.Linear
        Linear layer for projecting the output of the attention mechanism.
    smax : nn.Softmax
        Softmax layer to convert attention scores into probabilities.
    ln : BIALayerNorm
        Layer normalization applied to the input tensors before the attention mechanism.
    drop : nn.Dropout
        Dropout layer applied to the attention scores and output.
    """

    def __init__(
        self,
        n_emb: int,
        n_head: int,
        h_size: int,
        d_rate: float,
        is_decoder: bool = False,
        device: str = "cpu",
    ) -> None:
        super(ResMHAtten, self).__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.h_size = h_size
        self.is_decoder = is_decoder
        self.device = device

        self.q_proj = nn.Linear(n_emb, n_head * h_size)
        self.kv_proj = nn.Linear(n_emb, n_head * (h_size + h_size))
        self.o_proj = nn.Linear(n_head * h_size, n_emb)

        self.smax = nn.Softmax(dim=-1)
        self.ln = ResMHAtten(n_emb)
        self.drop = nn.Dropout(d_rate)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, qT, _ = q.shape
        _, kvT, _ = kv.shape

        # is self atten or cross atten
        is_self = torch.equal(kv, q)

        # fork for res con
        oq = q.clone()

        # pre-norm (diff than original tranfsormer)
        q = self.ln(q)
        kv = self.ln(kv) if is_self else kv

        # q, k, v --> qp, kp, vp    [B, xT, C]
        qp = self.q_proj(q)
        kp, vp = self.kv_proj(kv).split(split_size=self.n_head * self.h_size, dim=-1)

        # qp, kp, vp ==> (B, xT, nh, h)
        qp = qp.contiguous().view(B, qT, self.n_head, self.h_size)
        kp = kp.contiguous().view(B, kvT, self.n_head, self.h_size)
        vp = vp.contiguous().view(B, kvT, self.n_head, self.h_size)

        # [B, xT, h, nh] ==> [B * nh, xT, h]
        qp = qp.permute(0, 2, 1, 3).contiguous().view(-1, qT, self.h_size)
        kp = kp.permute(0, 2, 1, 3).contiguous().view(-1, kvT, self.h_size)
        vp = vp.permute(0, 2, 1, 3).contiguous().view(-1, kvT, self.h_size)

        # [B * nh, qT, h]   x   [B * nh, h, kvT]   ==>   [B * nh, qT, kvT]
        attn = torch.bmm(qp, kp.permute(0, 2, 1))
        attn = (1.0 / math.sqrt(self.h_size)) * attn  # /sqrt(h)

        # self atten mask
        if self.is_decoder and is_self:
            hide_future_toks = torch.ones_like(attn).tril().bool().to(self.device)
            attn = attn.masked_fill(~hide_future_toks, -float("inf"))

        attn = self.drop(self.smax(attn))

        # [B * nh, qT, kvT]   x   [B * nh, kvT, h]    ==>    [B * nh, qT, h]
        attn = torch.bmm(attn, vp)  # [B * nh, qT, h]
        attn = (
            attn.contiguous().view(B, self.n_head, qT, self.h_size).permute(0, 2, 1, 3)
        )  # [B, qT, nh, h]
        attn = attn.contiguous().view(B, qT, -1)  # [B, qT, nh * h]
        attn = self.o_proj(attn)  # [B, qT, e]

        out = self.drop(attn) + oq
        return out
