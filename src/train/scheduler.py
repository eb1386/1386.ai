# cosine lr schedule

import math


class CosineScheduler:
    def __init__(
        self,
        learning_rate: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        self.lr = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)
