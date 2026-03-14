# causal attention

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rotary_emb


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # expand kv heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, S, self.head_dim)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )

        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)
