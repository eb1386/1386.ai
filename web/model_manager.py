# model loading and switching

import gc
import re
import threading
from pathlib import Path

import torch
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config
from src.inference.template import build_prompt_ids, stop_sequences, penalty_exclude

ROOT = Path(__file__).resolve().parent.parent

MODEL_REGISTRY = {
    "plasma-1.0": {
        "name": "Plasma 1.0",
        "config": ROOT / "configs" / "finetune_1.0.yaml",
        "checkpoint": ROOT / "checkpoints" / "finetune_1.0_final.pt",
        "tokenizer": ROOT / "data" / "tokenizer_1.0.model",
        "params": "235M",
        "multiturn": False,
    },
    # 1.1 = the v4 audit sft (user's call, aug 17). template v4: bos at
    # conversation start, eos supervised after every turn. the v3 final and
    # pre-rebuild checkpoints stay on disk as baselines, just not listed.
    # no tokenizer fallback: 48k vocab; the 32k tokenizer would produce
    # garbage from embedding rows 32k-48k
    "plasma-1.1": {
        "name": "Plasma 1.1",
        "config": ROOT / "configs" / "finetune_1.1_v4.yaml",
        "checkpoint": ROOT / "checkpoints" / "finetune_1.1_v4_final.pt",
        "tokenizer": ROOT / "data" / "tokenizer_1.1.model",
        "params": "521M",
        "multiturn": True,
        "template": "v4",
    },
}

# private/experimental models live in an untracked local file:
#   web/models_local.json  ->  {"id": {"name", "config", "checkpoint",
#                               "tokenizer", "params", "multiturn", "template"}}
_local = ROOT / "web" / "models_local.json"
if _local.exists():
    import json as _json
    try:
        for _id, _info in _json.loads(_local.read_text(encoding="utf-8")).items():
            for _k in ("config", "checkpoint", "tokenizer"):
                if _k in _info:
                    _info[_k] = ROOT / _info[_k]
            MODEL_REGISTRY[_id] = _info
    except Exception as _e:
        print(f"[warn] models_local.json ignored: {_e}")

# serving defaults; audit-tuned (rp 1.3 caused progressive drift, and
# no_repeat_ngram kills degenerate loops the softer penalty lets through)
GEN_PARAMS = dict(temperature=0.5, top_k=30, top_p=0.85,
                  repetition_penalty=1.1, penalty_window=128,
                  no_repeat_ngram=4)
MAX_TOKENS = 300

# scraped-article template slots leaking from pretrain data
PLACEHOLDER_RE = re.compile(r"\[(?:Insert|Your|Name of)[^\]\n]{0,80}\]")

# invented-speaker cut: only ever cut on words that mean a dialogue turn.
# any other "Word:" (Method:, Python:, Notes:) is legitimate content.
ROLE_WORDS = (
    "user", "assistant", "system", "human", "boy", "girl", "man",
    "woman", "teacher", "student", "interviewer", "narrator", "doctor", "bot",
    "speaker", "customer", "agent", "mom", "dad",
    "friend", "host", "guest", "child", "kid", "reporter",
)
INLINE_SPEAKER_RE = re.compile(
    r"(?<=[.!?…])\s+(" + "|".join(ROLE_WORDS) + r")\s?:\s", re.IGNORECASE)
LINE_SPEAKER_RE = re.compile(
    r"\n[ \t]*(?:" + "|".join(ROLE_WORDS) + r"|q|a)[ \t]?:[ \t]", re.IGNORECASE
)


class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_id = None
        self.model = None
        self.model_cfg = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._load_tokenizer()

    def _load_tokenizer(self, model_id=None):
        tok_path = None
        if model_id and model_id in MODEL_REGISTRY:
            info = MODEL_REGISTRY[model_id]
            if info.get("tokenizer") and info["tokenizer"].exists():
                tok_path = info["tokenizer"]
            elif info.get("tokenizer_fallback") and info["tokenizer_fallback"].exists():
                tok_path = info["tokenizer_fallback"]
            else:
                # called with explicit model_id: this MUST find a tokenizer.
                # for plasma-1.1 the 32k tokenizer would silently corrupt output,
                # so fail loudly instead.
                raise FileNotFoundError(
                    f"tokenizer for {model_id} not found at {info.get('tokenizer')}"
                )
        if tok_path is None:
            # startup with no model selected: best-effort load 1.0 if present
            tok_path = ROOT / "data" / "tokenizer_1.0.model"
            if not tok_path.exists():
                self.tokenizer = None
                return
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
        # validate checkpoint shapes before loading to surface mismatches early
        ckpt = torch.load(str(info["checkpoint"]), map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        emb_w = sd.get("tok_emb.weight")
        if emb_w is not None:
            v, h = emb_w.shape
            if v != self.model_cfg.vocab_size or h != self.model_cfg.hidden_size:
                raise RuntimeError(
                    f"checkpoint/config shape mismatch for {model_id}: "
                    f"checkpoint embed=({v},{h}) vs config "
                    f"vocab={self.model_cfg.vocab_size}, hidden={self.model_cfg.hidden_size}"
                )
        self.model.load_state_dict(sd)
        self.model.eval()
        self.current_model_id = model_id
        self._load_tokenizer(model_id)

        print(f"Loaded {info['name']} ({self.model.count_parameters():,} params) on {self.device}")

    def generate(self, model_id, prompt, max_tokens=MAX_TOKENS, history=None, **overrides):
        # one request at a time: load_model swaps model AND tokenizer, so a
        # concurrent request on another model_id would generate with a
        # mismatched pair (1.0 is 32k vocab, 1.1 is 48k)
        with self._lock:
            self.load_model(model_id)

            from src.inference.generate import generate as gen_fn

            sp = self.tokenizer
            info = MODEL_REGISTRY.get(model_id, {})
            multiturn = info.get("multiturn", False)
            v4 = info.get("template") == "v4"
            budget = self.model_cfg.max_seq_len - max_tokens
            ids = build_prompt_ids(
                sp, prompt,
                history=history if multiturn else None,
                max_prompt_tokens=max(budget, 64),
                eos_between_turns=v4, with_bos=v4,
            )

            params = dict(GEN_PARAMS)
            params.update(overrides)
            output, meta = gen_fn(
                self.model, sp, "",
                prompt_ids=ids,
                max_tokens=max_tokens,
                device=self.device,
                return_new_only=True,
                return_meta=True,
                penalty_exclude=penalty_exclude(sp),
                stop_id_seqs=stop_sequences(sp),
                **params,
            )

        response = output.strip()

        for stop in ["\nUser:", "\n User:", "\nAssistant:", "\n Assistant:",
                     "\nSystem:", "\nHuman:", "\nQuestion:", "\n\n\n"]:
            if stop in response:
                response = response[:response.index(stop)]

        response = self._cut_dialogue(response)
        response = self._clean_response(response.strip(), meta["stop"])
        return response or "(empty response)"

    @staticmethod
    def _cut_dialogue(text):
        """drop an invented speaker turn and everything after it"""
        cut = len(text)
        m = INLINE_SPEAKER_RE.search(text)
        if m:
            cut = min(cut, m.start())
        m = LINE_SPEAKER_RE.search(text)
        if m:
            cut = min(cut, m.start())
        return text[:cut].rstrip()

    @staticmethod
    def _clean_response(text, stop_reason="length"):
        if not text:
            return text

        # drop scraped template placeholders, then trim per line
        text = PLACEHOLDER_RE.sub("", text)
        lines = text.split("\n")
        text = "\n".join(line.rstrip() for line in lines).strip()

        # a clean stop means the model finished on purpose; only a length
        # cutoff leaves a dangling fragment worth trimming. never trim when
        # the tail is a list item, header, or code fence.
        if stop_reason in ("length", "context") and len(text) > 100 \
                and text[-1] not in ".!?\"')":
            tail = text.rsplit("\n", 1)[-1].lstrip()
            structural = (tail.startswith(("-", "*", "```"))
                          or re.match(r"^\d+[\.\)]", tail) or tail.endswith(":"))
            if not structural:
                last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
                if last_period > 50:
                    text = text[:last_period + 1]

        return text
