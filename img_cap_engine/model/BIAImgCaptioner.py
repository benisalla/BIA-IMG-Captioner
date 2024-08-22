from typing import Dict
from img_cap_engine.model.Decoder import Decoder
from img_cap_engine.model.Encoder import Encoder
import torch
from torch import nn
import torch.nn.functional as F
import math


class BIAImgCaptioner(nn.Module):
    """
    BIAImgCaptioner is a neural network model designed for generating captions or descriptions from images.
    It consists of an encoder to process the input image and a decoder to generate text sequences based on
    the encoded image features. The model can be trained to generate accurate and contextually relevant
    captions for images.

    Parameters:
    -----------
    v_size (int):
        The size of the vocabulary used by the decoder.
    n_emb (int):
        The dimensionality of the embeddings.
    n_head (int): 
        The number of attention heads in the multi-head attention mechanism.
    h_size (int): 
        The size of the hidden layers.
    n_block (int): 
        The number of transformer blocks in both the encoder and decoder.
    exp_fac (int): 
        The expansion factor for the feedforward layers.
    max_seq_len (int): 
        The maximum sequence length for the generated captions.
    d_rate (float): 
        The dropout rate used in various layers.
    p_size (int): 
        The patch size for the image input.
    im_size (int): 
        The input image size.
    c_dim (int): 
        The number of channels in the input image.
    device (str): 
        The device on which the model is executed (e.g., 'cpu' or 'cuda').
    """

    def __init__(
        self,
        v_size: int = 10000,
        n_emb: int = 512,
        n_head: int = 8,
        h_size: int = 64,
        n_block: int = 6,
        exp_fac: int = 4,
        max_seq_len: int = 10000,
        d_rate: float = 0.0,
        p_size: int = 16,
        im_size: int = 64,
        c_dim: int = 3,
        device: str = "cpu",
    ):
        super(BIAImgCaptioner, self).__init__()

        self.v_size = v_size
        self.n_emb = n_emb
        self.n_head = n_head
        self.h_size = h_size
        self.n_block = n_block
        self.exp_fac = exp_fac
        self.max_seq_len = max_seq_len
        self.d_rate = d_rate
        self.p_size = p_size
        self.im_size = im_size
        self.c_dim = c_dim
        self.device = device

        self.encoder = Encoder(
            n_block=n_block,
            n_emb=n_emb,
            n_head=n_head,
            h_size=h_size,
            p_size=p_size,
            im_size=im_size,
            c_dim=c_dim,
            exp_fac=exp_fac,
            d_rate=d_rate,
            device=device,
        )

        self.decoder = Decoder(
            v_size=v_size,
            n_emb=n_emb,
            n_head=n_head,
            h_size=h_size,
            max_seq_len=max_seq_len,
            n_block=n_block,
            exp_fac=exp_fac,
            d_rate=d_rate,
            device=device,
        )

        self.decoder.fc.weight = self.decoder.pte.weight
        self.init_weights()
        self.to(device)

    def init_weights(self) -> None:
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

    def forward(self, img: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ex = self.encoder(img=img)
        logs = self.decoder(idx=idx, ex=ex)
        return logs

    def get_init_args(self) -> Dict[str, int]:
        return {
            "v_size": self.v_size,
            "n_emb": self.n_emb,
            "n_head": self.n_head,
            "h_size": self.h_size,
            "n_block": self.n_block,
            "exp_fac": self.exp_fac,
            "max_seq_len": self.max_seq_len,
            "d_rate": self.d_rate,
            "p_size": self.p_size,
            "im_size": self.im_size,
            "c_dim": self.c_dim,
            "device": self.device,
        }

    @torch.no_grad()
    def describe(
        self,
        img: torch.Tensor,
        tokenizer,
        temperature: float = 1.0,
        beam_size: int = 4,
        len_norm_coeff: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        is_ltr: bool = False,
        max_beam_fork: int = 128,
    ):
        device = self.device
        self = self.to(device)
        self.eval()

        k = beam_size  # Beam size
        n_hypo = min(k, 10)  # n of hypo
        vs = self.v_size  # Vocab size

        img = img.to(device)
        ex = self.encoder(img=img)

        hypo = torch.LongTensor([[tokenizer.bos_token_id]]).to(device)  # d_idx: <SOS>
        hypo_scores = torch.zeros(1).to(device)  # 1 score

        com_hypo = list()
        com_hypo_scores = list()

        step = 1
        while True:
            s = hypo.size(0)  # s
            logits = self.decoder(
                idx=hypo,
                ex=ex.repeat(s, 1, 1),
            )  # [s, step, vs]
            flogits = self.top_k_top_p_filtering(
                logits=logits[:, -1, :], top_k=top_k, top_p=top_p
            )
            scores = flogits / max(temperature + 1.0, 1e-8)  # [s, vs]
            scores = F.log_softmax(scores, dim=-1)  # [s, vs]
            scores = hypo_scores.unsqueeze(1) + scores  # prev scores + curr scores

            top_k_hypo_scores, fttn_idx = scores.view(-1).topk(
                k, 0, True, True
            )  # top(vs) = k

            prev_tok_idx = fttn_idx // vs  # prev [k]
            next_tok_idx = fttn_idx % vs  # next [k]

            top_k_hypo = torch.cat(
                [hypo[prev_tok_idx], next_tok_idx.unsqueeze(1)], dim=1
            )  # [k, step + 1]

            complete = next_tok_idx == tokenizer.eos_token_id  # <EOS>? : [k], bool

            com_hypo.extend(top_k_hypo[complete].tolist())
            norm = math.pow(
                ((5 + step) / (5 + 1)), len_norm_coeff
            )  # chance(long sentence ~ short sentence)
            com_hypo_scores.extend((top_k_hypo_scores[complete] / norm).tolist())

            if len(com_hypo) >= n_hypo:
                break  # enough hypos

            hypo = top_k_hypo[~complete]  # [s, step + 1] incomplete hypos
            hypo_scores = top_k_hypo_scores[~complete]  # d_idx(s): [s]

            if step > max_beam_fork:
                break  # stop if no EOS is found
            step += 1

        if len(com_hypo) == 0:  # in case no EOS is found
            com_hypo = hypo.tolist()
            com_hypo_scores = hypo_scores.tolist()

        all_hypos = list()  # idx ==> string
        spec_toks = [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]
        if is_ltr:
            com_hypo = [
                tokenizer.decode([t for t in s if t not in spec_toks])[::-1]
                for s in com_hypo
            ]
        else:
            com_hypo = [
                tokenizer.decode([t for t in s if t not in spec_toks]) for s in com_hypo
            ]

        for i, h in enumerate(com_hypo):
            all_hypos.append({"hypothesis": h, "score": com_hypo_scores[i]})

        max_idx = com_hypo_scores.index(max(com_hypo_scores))
        best_hypo = all_hypos[max_idx]["hypothesis"]

        return best_hypo, all_hypos

    def top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        return logits
