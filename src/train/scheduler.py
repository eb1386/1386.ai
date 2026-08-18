# lr schedules

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


class WSDScheduler:
    """warmup-stable-decay (trapezoidal).

    holds a constant lr through the bulk of training and only cools down over
    the final `decay_steps`. that means the run length does not have to be
    committed up front: you can cool down from any point and get a usable
    model, or keep going. the (1 - sqrt) cooldown shape is the one reported to
    match or beat cosine at equal token budget.
    """

    def __init__(
        self,
        learning_rate: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
        decay_steps: int | None = None,
        decay_frac: float = 0.1,
        shape: str = "1-sqrt",
    ):
        self.lr = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.decay_steps = int(decay_steps) if decay_steps else max(1, int(max_steps * decay_frac))
        self.decay_start = max(warmup_steps, max_steps - self.decay_steps)
        self.shape = shape

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        if step < self.decay_start:
            return self.lr
        if step >= self.max_steps:
            return self.min_lr
        frac = (step - self.decay_start) / max(1, self.max_steps - self.decay_start)
        if self.shape == "linear":
            coeff = 1.0 - frac
        elif self.shape == "cosine":
            coeff = 0.5 * (1.0 + math.cos(math.pi * frac))
        else:  # 1-sqrt
            coeff = 1.0 - math.sqrt(frac)
        return self.min_lr + coeff * (self.lr - self.min_lr)


def build_scheduler(train_cfg: dict):
    """pick the schedule named in the config (default cosine)."""
    kind = str(train_cfg.get("lr_schedule", "cosine")).lower()
    common = dict(
        learning_rate=train_cfg["learning_rate"],
        min_lr=train_cfg["min_lr"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
    )
    if kind in ("wsd", "trapezoid", "trapezoidal"):
        return WSDScheduler(
            **common,
            decay_steps=train_cfg.get("decay_steps"),
            decay_frac=float(train_cfg.get("decay_frac", 0.1)),
            shape=str(train_cfg.get("decay_shape", "1-sqrt")),
        )
    return CosineScheduler(**common)
