# model config

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 8
    intermediate_size: int = 1376
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    def param_count_estimate(self) -> int:
        embed = self.vocab_size * self.hidden_size * 2  # embed + output
        attn = self.num_layers * (
            self.hidden_size * self.hidden_size  # Q
            + self.hidden_size * (self.head_dim * self.num_kv_heads) * 2  # K, V
            + self.hidden_size * self.hidden_size  # O
        )
        ffn = self.num_layers * (
            self.hidden_size * self.intermediate_size * 3  # gate, up, down
        )
        norm = self.num_layers * self.hidden_size * 2 + self.hidden_size
        return embed + attn + ffn + norm

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
