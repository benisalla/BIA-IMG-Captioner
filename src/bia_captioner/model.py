"""Image -> caption model.

Big picture:
    image --encoder--> N "image tokens" --projection + position + LayerNorm--> memory
    caption tokens --Transformer decoder (cross-attends to memory)--> next-token logits

Image encoders (`ModelConfig.encoder`; all but bia-vit can be pretrained or from scratch):

    "bia-vit"                 BIA Vision Transformer (hand-written), always from scratch
    "vit_b_16" ... "vit_h_14" torchvision Vision Transformers (ImageNet weights)
    "resnet18" ... "resnet152" torchvision ResNets (ImageNet weights)
    "hf:<model id>"           any Hugging Face vision backbone (ViT-21k, DINOv2, CLIP, ...)

The decoder is a hand-written pre-norm Transformer (masked self-attention,
cross-attention to the image tokens, feed-forward), GPT-2 style.
"""

import inspect
import math

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F

from bia_captioner.config import (
    BIA_VIT,
    HF_PREFIX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TORCHVISION_VITS,
    ModelConfig,
)
from bia_captioner.tokenizer import IGNORE_INDEX

INIT_STD = 0.02  # GPT-2 style init for every new (non-pretrained) weight
RESNET_STRIDE = 32  # a ResNet shrinks the image 32x before its final pooling


# ---------------------------------------------------------------------------
# Building blocks (shared by the BIA-ViT encoder and the decoder)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention. Self-attention when `context` is None, else cross-attention.

    Uses PyTorch's fused scaled_dot_product_attention (FlashAttention /
    memory-efficient kernels on GPU).
    """

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)  # keys and values in one matmul
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, context: Tensor | None = None, causal: bool = False) -> Tensor:
        """x [B, T, d] attends to context [B, S, d] (or to itself) -> [B, T, d]."""
        context = x if context is None else context
        batch, t_len, d_model = x.shape
        s_len = context.shape[1]
        head_dim = d_model // self.heads

        # [B, T, d] -> [B, heads, T, head_dim]
        q = self.q_proj(x).view(batch, t_len, self.heads, head_dim).transpose(1, 2)
        kv = self.kv_proj(context).view(batch, s_len, 2, self.heads, head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)  # each [B, heads, S, head_dim]

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=causal
        )
        return self.out_proj(out.transpose(1, 2).reshape(batch, t_len, d_model))


class FeedForward(nn.Module):
    """Position-wise MLP: d -> mult*d -> d with GELU."""

    def __init__(self, d_model: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, d_model: int, heads: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mult, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: masked self-attention, cross-attention, feed-forward."""

    def __init__(self, d_model: int, heads: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mult, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        # Causal mask: token t only sees tokens 0..t. Right padding comes after
        # every real token, so real tokens never see padding either.
        x = x + self.dropout(self.self_attn(self.norm1(x), causal=True))
        # The memory is already normalised (image_norm), so only x is normed here.
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=memory))
        return x + self.dropout(self.mlp(self.norm3(x)))


# ---------------------------------------------------------------------------
# Image encoders: images [B, 3, H, W] -> features [B, N, C]
# ---------------------------------------------------------------------------


class BIAViT(nn.Module):
    """The BIA Vision Transformer, trained from scratch.

    Patches of `patch_size` pixels -> linear embedding (a strided conv) ->
    learned position -> pre-norm Transformer blocks -> final LayerNorm.
    """

    def __init__(self, config: ModelConfig, image_size: int) -> None:
        super().__init__()
        if image_size % config.patch_size:
            raise ValueError(
                f"image_size {image_size} must be a multiple of patch_size {config.patch_size}"
            )
        d_model = config.d_model
        self.num_tokens = (image_size // config.patch_size) ** 2
        self.out_dim = d_model
        # A conv with kernel = stride = patch size embeds each patch independently.
        self.patch_embedding = nn.Conv2d(
            3, d_model, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.position = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, config.attention_heads, config.ffn_mult, config.dropout)
            for _ in range(config.encoder_layers)
        )
        # Pre-norm blocks leave the output un-normalised, so add a final LayerNorm.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pixel_values: Tensor) -> Tensor:
        x = self.patch_embedding(pixel_values)  # [B, d, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d]
        x = self.dropout(x + self.position)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TorchvisionViT(nn.Module):
    """A torchvision Vision Transformer without its classifier: all token features.

    pretrained=True loads the default ImageNet weights, which fix the image size
    (224 for vit_b_16 ... vit_l_32, 518 for vit_h_14).
    """

    def __init__(self, name: str, image_size: int, pretrained: bool) -> None:
        super().__init__()
        builder = getattr(torchvision.models, name)
        if pretrained:
            weights = torchvision.models.get_model_weights(name).DEFAULT
            native = weights.meta["min_size"][0]
            if image_size != native:
                raise ValueError(f"pretrained {name} expects --image-size {native}")
            self.vit = builder(weights=weights)
        else:
            self.vit = builder(weights=None, image_size=image_size)
        if image_size % self.vit.patch_size:
            raise ValueError(f"image_size must be a multiple of {self.vit.patch_size} for {name}")
        self.vit.heads = nn.Identity()  # drop the ImageNet classifier
        self.num_tokens = (image_size // self.vit.patch_size) ** 2 + 1  # patches + class token
        self.out_dim = self.vit.hidden_dim

    def forward(self, pixel_values: Tensor) -> Tensor:
        x = self.vit._process_input(pixel_values)  # patch embedding [B, N, width]
        class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        return self.vit.encoder(x)  # adds positions, runs the blocks, final LayerNorm


class TorchvisionResNet(nn.Module):
    """A torchvision ResNet without pooling/classifier: one token per grid cell."""

    def __init__(self, name: str, image_size: int, pretrained: bool) -> None:
        super().__init__()
        backbone = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
        self.out_dim = backbone.fc.in_features  # 512 for resnet18/34, 2048 for 50/101/152
        self.body = nn.Sequential(*list(backbone.children())[:-2])  # keep the spatial grid
        self.num_tokens = math.ceil(image_size / RESNET_STRIDE) ** 2  # 224px -> 7x7 = 49

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.body(pixel_values).flatten(2).transpose(1, 2)  # [B, h*w, C]


class HuggingFaceEncoder(nn.Module):
    """Any Hugging Face vision backbone; its last hidden states are the image tokens.

    CLIP / SigLIP checkpoints contain a text tower too: only the image tower is
    kept. Parameters that never influence the tokens (poolers, projection
    heads) are found with one probe pass and frozen, so DistributedDataParallel
    does not wait for gradients that never come.
    """

    def __init__(self, model_id: str, image_size: int, pretrained: bool) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if pretrained:
            model = AutoModel.from_pretrained(model_id)
        else:
            model = AutoModel.from_config(AutoConfig.from_pretrained(model_id))
        self.model = getattr(model, "vision_model", model)  # CLIP / SigLIP: image tower
        config = self.model.config
        self.out_dim = getattr(config, "hidden_size", None) or config.hidden_sizes[-1]
        accepts = inspect.signature(self.model.forward).parameters
        # ViT-style models need this to run at a size other than their native one.
        self.extra = (
            {"interpolate_pos_encoding": True} if "interpolate_pos_encoding" in accepts else {}
        )

        # Probe pass: number of tokens, and which parameters get no gradient.
        tokens = self(torch.zeros(1, 3, image_size, image_size))
        self.num_tokens = tokens.shape[1]
        tokens.float().sum().backward()
        self.unused_parameters = set()
        for name, param in self.named_parameters():
            if param.grad is None:
                self.unused_parameters.add(name)
                param.requires_grad_(False)
            param.grad = None

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden = self.model(pixel_values=pixel_values, **self.extra).last_hidden_state
        if hidden.dim() == 4:  # convolutional backbone: [B, C, h, w] -> [B, h*w, C]
            hidden = hidden.flatten(2).transpose(1, 2)
        return hidden


def build_encoder(config: ModelConfig, image_size: int, pretrained: bool) -> nn.Module:
    """The image encoder named by config.encoder."""
    if config.encoder == BIA_VIT:
        return BIAViT(config, image_size)  # always from scratch
    if config.encoder.startswith(HF_PREFIX):
        return HuggingFaceEncoder(config.encoder[len(HF_PREFIX) :], image_size, pretrained)
    if config.encoder in TORCHVISION_VITS:
        return TorchvisionViT(config.encoder, image_size, pretrained)
    return TorchvisionResNet(config.encoder, image_size, pretrained)


def encoder_normalization(config: ModelConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) the encoder's inputs must be normalised with.

    torchvision models and BIA-ViT use ImageNet statistics; a Hugging Face
    model brings its own (CLIP and ViT-21k differ from ImageNet).
    """
    if config.encoder.startswith(HF_PREFIX):
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(config.encoder[len(HF_PREFIX) :])
        mean, std = getattr(processor, "image_mean", None), getattr(processor, "image_std", None)
        if mean is not None and std is not None:
            return tuple(mean), tuple(std)
    return IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# The captioner
# ---------------------------------------------------------------------------


class ImageCaptioner(nn.Module):
    """Image encoder + autoregressive Transformer decoder."""

    def __init__(
        self, config: ModelConfig, image_size: int = 224, pretrained_encoder: bool | None = None
    ) -> None:
        """pretrained_encoder: None = follow config.pretrained. Loading a checkpoint
        passes False: its weights replace the pretrained ones anyway."""
        super().__init__()
        self.config = config
        if pretrained_encoder is None:
            pretrained_encoder = config.pretrained
        self.encoder_frozen = False
        d_model = config.d_model

        # ---- Image encoder -------------------------------------------------
        self.encoder = build_encoder(config, image_size, pretrained_encoder)
        self.num_image_tokens = self.encoder.num_tokens
        # Map encoder features to the decoder width.
        self.image_projection = nn.Linear(self.encoder.out_dim, d_model)
        # Attention ignores order, so each image token gets a learned "where am I" vector.
        self.image_position = nn.Parameter(torch.zeros(1, self.num_image_tokens, d_model))
        # Normalise the memory so its scale matches what the decoder expects.
        self.image_norm = nn.LayerNorm(d_model)

        # ---- Text decoder --------------------------------------------------
        # padding_idx: the input lookup never sends gradient to the pad row (the tied
        # output head still does: it learns that pad is never the next token).
        self.token_embedding = nn.Embedding(
            config.vocab_size, d_model, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.max_tokens, d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, config.attention_heads, config.ffn_mult, config.dropout)
            for _ in range(config.decoder_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        # Output head shares its weights with the token embedding ("weight tying"):
        # fewer parameters and usually a better language model.
        self.output = nn.Linear(d_model, config.vocab_size, bias=False)

        self._init_new_weights()
        self.output.weight = self.token_embedding.weight  # tie after init

    # ------------------------------------------------------------------ init
    def _init_new_weights(self) -> None:
        """N(0, 0.02) for new Linear/Conv/Embedding weights, zero biases.

        Residual output projections get a smaller std (GPT-2): each block adds
        to the residual stream, so this keeps its variance from growing with depth.
        Library encoders (torchvision / Hugging Face) keep their own weights,
        pretrained or freshly initialised by the library; LayerNorms keep (1, 0).
        """
        cfg = self.config
        skip_encoder = not isinstance(self.encoder, BIAViT)
        for name, module in self.named_modules():
            in_encoder = name.startswith("encoder")
            if skip_encoder and in_encoder:
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                std = INIT_STD
                if name.endswith(("out_proj", "fc2")):  # writes into the residual stream
                    depth = cfg.encoder_layers if in_encoder else cfg.decoder_layers
                    std = INIT_STD / math.sqrt(2 * depth)
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.image_position, mean=0.0, std=INIT_STD)
        if isinstance(self.encoder, BIAViT):
            nn.init.normal_(self.encoder.position, mean=0.0, std=INIT_STD)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_token_id].zero_()  # pad starts at zero

    # -------------------------------------------------------------- freezing
    def freeze_encoder(self, frozen: bool = True) -> None:
        """Freeze (or unfreeze) the image encoder.

        A frozen encoder gets no gradients, runs without autograd (faster, less
        memory) and stays in eval mode so BatchNorm/dropout behave as at inference.
        """
        self.encoder_frozen = frozen
        never_trained = getattr(self.encoder, "unused_parameters", set())
        for name, param in self.encoder.named_parameters():
            param.requires_grad_(not frozen and name not in never_trained)
        self.train(self.training)

    def train(self, mode: bool = True) -> "ImageCaptioner":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    # --------------------------------------------------------------- encoder
    def encode(self, pixel_values: Tensor) -> Tensor:
        """Images [B, 3, H, W] -> memory [B, N, d_model]."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.encoder_frozen):
            features = self.encoder(pixel_values)  # [B, N, C]
        if features.shape[1] != self.num_image_tokens:
            raise ValueError(
                f"encoder produced {features.shape[1]} image tokens but the model was built "
                f"for {self.num_image_tokens}; pass the right image_size to the constructor"
            )
        memory = self.image_projection(features) + self.image_position
        return self.image_norm(memory)

    # --------------------------------------------------------------- decoder
    def decode(self, memory: Tensor, input_ids: Tensor) -> Tensor:
        """Memory + caption tokens [B, T] -> logits [B, T, vocab]."""
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_tokens:
            raise ValueError(f"sequence length {seq_len} exceeds max_tokens")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x, memory)
        return self.output(self.final_norm(x))

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        labels: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> tuple[Tensor, Tensor | None]:
        """Return (logits, loss). Loss is None when no labels are given.

        `labels` is aligned with `input_ids` (same shape); padding is -100.
        We shift inside: the logits at position t predict the label at t + 1.
        """
        logits = self.decode(self.encode(pixel_values), input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
                label_smoothing=label_smoothing,
            )
        return logits, loss

    # ------------------------------------------------------------ generation
    def _forbid_special_tokens(self, logits: Tensor) -> Tensor:
        """Pad (and BOS, when it is not also EOS) are never real predictions."""
        logits[:, self.config.pad_token_id] = float("-inf")
        if self.config.bos_token_id != self.config.eos_token_id:
            logits[:, self.config.bos_token_id] = float("-inf")
        return logits

    def _max_new_tokens(self, max_new_tokens: int | None) -> int:
        limit = self.config.max_tokens - 1  # BOS already uses one slot
        return limit if max_new_tokens is None else min(max_new_tokens, limit)

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Tensor,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Tensor:
        """Caption each image, one token at a time (greedy or sampling).

        Returns [B, T] token ids starting with BOS. Rows that already produced
        EOS are filled with pad. temperature=0 means greedy (always the top token).
        Train/eval mode is left to the caller (call model.eval() first).
        """
        cfg = self.config
        memory = self.encode(pixel_values)  # image is encoded once and reused
        batch = pixel_values.shape[0]
        device = pixel_values.device
        tokens = torch.full((batch, 1), cfg.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(self._max_new_tokens(max_new_tokens)):
            logits = self.decode(memory, tokens)[:, -1].float()  # only the last position matters
            self._forbid_special_tokens(logits)
            next_token = _sample(logits, temperature, top_k, top_p)
            next_token = next_token.masked_fill(finished, cfg.pad_token_id)  # done rows get pad
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            finished |= next_token.eq(cfg.eos_token_id)
            if finished.all():
                break
        return tokens

    @torch.no_grad()
    def beam_search(
        self,
        pixel_values: Tensor,
        beam_size: int = 3,
        max_new_tokens: int | None = None,
        length_penalty: float = 0.6,
    ) -> list[tuple[Tensor, float]]:
        """Beam search for ONE image [1, 3, H, W].

        Keeps the `beam_size` most likely partial captions at each step. Scores
        are summed log-probabilities divided by the GNMT length penalty
        ((5 + length) / 6) ** length_penalty, so short captions are not unfairly
        preferred (0 = no normalisation). Returns up to `beam_size`
        (token ids starting with BOS, score) pairs, best first.
        """
        if pixel_values.shape[0] != 1:
            raise ValueError("beam_search takes one image; loop over images for a batch")
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")
        cfg = self.config
        memory = self.encode(pixel_values)
        beams = torch.full((1, 1), cfg.bos_token_id, dtype=torch.long, device=memory.device)
        scores = torch.zeros(1, device=memory.device)
        finished: list[tuple[list[int], float]] = []

        def normalised(score: float, length: int) -> float:
            return score / (((5 + length) / 6) ** length_penalty)

        for step in range(1, self._max_new_tokens(max_new_tokens) + 1):
            logits = self.decode(memory.expand(beams.shape[0], -1, -1), beams)[:, -1].float()
            log_probs = self._forbid_special_tokens(logits).log_softmax(dim=-1)
            candidates = (scores[:, None] + log_probs).flatten()  # every beam x every token
            # 2 x beam_size candidates: enough to keep beam_size alive even if some end here.
            top_scores, top_ids = candidates.topk(min(2 * beam_size, candidates.numel()))
            rows, next_tokens = top_ids // cfg.vocab_size, top_ids % cfg.vocab_size

            alive: list[list[int]] = []
            alive_scores: list[float] = []
            for score, row, token in zip(top_scores.tolist(), rows.tolist(), next_tokens.tolist()):
                sequence = beams[row].tolist() + [token]
                if token == cfg.eos_token_id:
                    finished.append((sequence, normalised(score, step)))
                else:
                    alive.append(sequence)
                    alive_scores.append(score)
                if len(alive) == beam_size:
                    break
            if len(finished) >= beam_size or not alive:
                break
            beams = torch.tensor(alive, device=memory.device)
            scores = torch.tensor(alive_scores, device=memory.device)
        else:
            # Hit the length limit: unfinished beams are candidates too.
            length = beams.shape[1] - 1
            finished += [
                (b, normalised(s, length)) for b, s in zip(beams.tolist(), scores.tolist())
            ]

        finished.sort(key=lambda item: item[1], reverse=True)
        return [(torch.tensor(seq), score) for seq, score in finished[:beam_size]]

    # ------------------------------------------------------------- optimizer
    def parameter_groups(self, weight_decay: float, lr: float, encoder_lr: float) -> list[dict]:
        """AdamW parameter groups.

        - encoder params use `encoder_lr`, everything else uses `lr`;
        - weight decay only on Linear / Conv weight matrices; biases, norms,
          embeddings and position/class tokens get none;
        - frozen parameters are left out.
        named_parameters() lists the tied output/embedding weight only once.
        """
        decay_ids = {
            id(module.weight)
            for module in self.modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        }
        decay_ids.discard(id(self.token_embedding.weight))  # tied with the output head

        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
            (is_enc, decay): [] for is_enc in (True, False) for decay in (True, False)
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            groups[(name.startswith("encoder."), id(param) in decay_ids)].append(param)

        return [
            {
                "name": "encoder" if is_encoder else "decoder",  # shown in the training log
                "params": params,
                "lr": encoder_lr if is_encoder else lr,
                "weight_decay": weight_decay if decay else 0.0,
            }
            for (is_encoder, decay), params in groups.items()
            if params  # skip empty groups (e.g. frozen encoder)
        ]


def _sample(logits: Tensor, temperature: float, top_k: int | None, top_p: float | None) -> Tensor:
    """Pick one token per row from logits [B, V]."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)  # greedy
    logits = logits / temperature  # <1 sharper, >1 more random
    if top_k is not None:
        # Keep only the k best tokens; the rest get probability 0.
        kth_best = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    if top_p is not None and top_p < 1.0:
        # Nucleus: keep the smallest set of best tokens whose probability sums to top_p.
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        drop = probs.cumsum(dim=-1) - probs > top_p  # mass *before* this token already >= p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


__all__ = [
    "IGNORE_INDEX",
    "BIAViT",
    "HuggingFaceEncoder",
    "ImageCaptioner",
    "TorchvisionResNet",
    "TorchvisionViT",
    "build_encoder",
    "encoder_normalization",
]
