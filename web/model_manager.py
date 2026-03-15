# model loading and switching

import gc
from pathlib import Path

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent

MODEL_REGISTRY = {
    "plasma-1.0": {
        "name": "Plasma 1.0",
        "config": ROOT / "configs" / "finetune_1.0.yaml",
        "checkpoint": ROOT / "checkpoints" / "finetune_1.0_final.pt",
        "tokenizer": ROOT / "data" / "tokenizer_v4.model",
        "params": "235M",
        "multiturn": False,
    },
    "plasma-1.1": {
        "name": "Plasma 1.1",
        "config": ROOT / "configs" / "finetune_1.1.yaml",
        "checkpoint": ROOT / "checkpoints" / "finetune_1.1_final.pt",
        "tokenizer": ROOT / "data" / "tokenizer_1.1.model",
        "tokenizer_fallback": ROOT / "data" / "tokenizer_v4.model",
        "params": "500M",
        "multiturn": True,
    },
}

MAX_HISTORY_TOKENS = 768


class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_id = None
        self.model = None
        self.model_cfg = None
        self.tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self, model_id=None):
        tok_path = None
        if model_id and model_id in MODEL_REGISTRY:
            info = MODEL_REGISTRY[model_id]
            if info.get("tokenizer") and info["tokenizer"].exists():
                tok_path = info["tokenizer"]
            elif info.get("tokenizer_fallback") and info["tokenizer_fallback"].exists():
                tok_path = info["tokenizer_fallback"]
        if tok_path is None:
            tok_path = ROOT / "data" / "tokenizer_v4.model"
        if tok_path.exists():
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tok_path))

    def get_available_models(self):
        models = []
        for model_id, info in MODEL_REGISTRY.items():
            available = info["checkpoint"].exists()
            models.append({
                "id": model_id,
                "name": info["name"],
                "params": info["params"],
                "available": available,
            })
        return models

    def load_model(self, model_id):
        if model_id == self.current_model_id and self.model is not None:
            return

        info = MODEL_REGISTRY.get(model_id)
        if not info:
            raise ValueError(f"Unknown model: {model_id}")
        if not info["checkpoint"].exists():
            raise FileNotFoundError(f"Checkpoint not found: {info['checkpoint']}")

        if self.model is not None:
            del self.model
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        cfg = load_config(str(info["config"]))
        self.model_cfg = ModelConfig.from_dict(cfg["model"])

        self.model = Transformer(self.model_cfg).to(self.device)
        load_checkpoint(str(info["checkpoint"]), self.model)
        self.model.eval()
        self.current_model_id = model_id
        self._load_tokenizer(model_id)

        print(f"Loaded {info['name']} ({self.model.count_parameters():,} params) on {self.device}")

    def _build_prompt(self, model_id, message, history=None):
        info = MODEL_REGISTRY.get(model_id, {})
        supports_multiturn = info.get("multiturn", False)

        if not supports_multiturn or not history:
            return f"User: {message}\nAssistant:"

        turns = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                turns.append(f"User: {content}")
            elif role == "assistant":
                turns.append(f"Assistant: {content}")

        turns.append(f"User: {message}")
        turns.append("Assistant:")

        full_prompt = "\n".join(turns)

        if self.tokenizer:
            tokens = self.tokenizer.encode(full_prompt, out_type=int)
            if len(tokens) > MAX_HISTORY_TOKENS:
                while len(turns) > 3 and len(tokens) > MAX_HISTORY_TOKENS:
                    turns.pop(0)
                    full_prompt = "\n".join(turns)
                    tokens = self.tokenizer.encode(full_prompt, out_type=int)

        return full_prompt

    def generate(self, model_id, prompt, max_tokens=200, temperature=0.15,
                 top_k=8, top_p=0.85, repetition_penalty=1.5, history=None):
        self.load_model(model_id)

        from src.inference.generate import generate as gen_fn
        full_prompt = self._build_prompt(model_id, prompt, history)

        output = gen_fn(
            self.model, self.tokenizer, full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=self.device,
        )

        response = output[len(full_prompt):].strip()

        for stop in ["\nUser:", "\nSystem:", "\nHuman:", "\nQuestion:", "\n\n\n"]:
            if stop in response:
                response = response[:response.index(stop)]

        response = self._clean_response(response.strip())
        return response or "(empty response)"

    @staticmethod
    def _clean_response(text):
        if not text:
            return text

        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cleaned.append(line)

        text = "\n".join(cleaned)

        stripped = text.strip().rstrip(".")
        if stripped.isdigit() and len(stripped) <= 3:
            return "I'm not sure about that."

        if len(text) > 100 and text[-1] not in ".!?\"'":
            last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_period > 50:
                text = text[:last_period + 1]

        return text
