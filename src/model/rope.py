# rotary embeddings
#
# real-valued formulation. the complex64 version (torch.polar / view_as_complex)
# is mathematically identical but torchinductor cannot codegen complex ops, so
# it breaks the compiled graph. this pairs adjacent dims (x0,x1), (x2,x3), ...
# exactly like view_as_complex did, so checkpoints stay compatible.

import torch


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """returns (max_seq_len, head_dim//2, 2) holding [cos, sin]."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    B, H, S, D = x.shape
    xf = x.float().reshape(B, H, S, D // 2, 2)
    x_even = xf[..., 0]
    x_odd = xf[..., 1]

    freqs = freqs_cis[:S]                      # (S, D//2, 2)
    cos = freqs[..., 0].unsqueeze(0).unsqueeze(0)
    sin = freqs[..., 1].unsqueeze(0).unsqueeze(0)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack([out_even, out_odd], dim=-1).reshape(B, H, S, D)
    return out.type_as(x)
