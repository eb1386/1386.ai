# training utilities

import json
import time
from pathlib import Path

import torch
import yaml

from src.model.config import ModelConfig


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    path: str,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0), ckpt.get("config", {})


class JSONLLogger:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = open(path, "a", encoding="utf-8")

    def log(self, data: dict):
        data["timestamp"] = time.time()
        self.file.write(json.dumps(data) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
