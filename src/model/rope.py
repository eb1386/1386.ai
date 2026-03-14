# rotary embeddings

import torch


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs_cis[:S].unsqueeze(0).unsqueeze(0)  # (1, 1, S, D//2)
    x_rot = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return x_rot.type_as(x)
