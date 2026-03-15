# transformer model

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .attention import CausalAttention
from .feed_forward import SwiGLUFFN
from .normalization import RMSNorm
from .rope import precompute_freqs_cis


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if use_cache:
            attn_out, new_cache = self.attn(
                self.attn_norm(x), freqs_cis, kv_cache=kv_cache, use_cache=True,
            )
            x = x + attn_out
            x = x + self.ffn(self.ffn_norm(x))
            return x, new_cache

        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.output.weight = self.tok_emb.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        B, S = tokens.shape
        x = self.tok_emb(tokens)
        freqs = self.freqs_cis[start_pos : start_pos + S]

        use_cache = kv_caches is not None

        if use_cache:
            new_caches = []
            for i, layer in enumerate(self.layers):
                layer_cache = kv_caches[i] if i < len(kv_caches) else None
                x, new_cache = layer(x, freqs, kv_cache=layer_cache, use_cache=True)
                new_caches.append(new_cache)
            x = self.norm(x)
            return self.output(x), new_caches

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs, use_reentrant=False)
            else:
                x = layer(x, freqs)

        x = self.norm(x)
        return self.output(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
