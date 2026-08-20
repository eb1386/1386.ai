#!/usr/bin/env python3
# export a plasma checkpoint to gguf (llama arch) for llama.cpp / wasm
#
#   python scripts/export_gguf.py --checkpoint checkpoints/finetune_1.1_v4_final.pt \
#       --config configs/finetune_1.1_v4.yaml --out dist/plasma-1.1-f16.gguf
#
# the model is llama-shaped (rmsnorm, interleaved-pair rope, gqa, swiglu,
# tied embeddings), so tensors map 1:1. rope here is the interleaved-pair
# ("NORM") style llama.cpp uses natively, so no q/k permutation is needed.

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.train.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/finetune_1.1_v4_final.pt")
    ap.add_argument("--config", default="configs/finetune_1.1_v4.yaml")
    ap.add_argument("--tokenizer", default="data/tokenizer_1.1.model")
    ap.add_argument("--out", default="dist/plasma-1.1-f16.gguf")
    ap.add_argument("--name", default="Plasma 1.1")
    args = ap.parse_args()

    import gguf
    import sentencepiece as spm

    cfg = load_config(str(ROOT / args.config))["model"]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.checkpoint} ...")
    sd = torch.load(str(ROOT / args.checkpoint), map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)

    head_dim = cfg["hidden_size"] // cfg["num_heads"]
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_name(args.name)
    w.add_context_length(cfg["max_seq_len"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(cfg["num_layers"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_heads"])
    w.add_head_count_kv(cfg["num_kv_heads"])
    w.add_rope_dimension_count(head_dim)
    w.add_rope_freq_base(float(cfg.get("rope_theta", 10000.0)))
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # tokenizer: sentencepiece pieces embedded directly
    sp = spm.SentencePieceProcessor()
    sp.load(str(ROOT / args.tokenizer))
    n = sp.get_piece_size()
    assert n == cfg["vocab_size"], (n, cfg["vocab_size"])
    tokens, scores, types = [], [], []
    for i in range(n):
        piece = sp.id_to_piece(i)
        tokens.append(piece.encode("utf-8"))
        scores.append(sp.get_score(i))
        if sp.is_unknown(i):
            t = gguf.TokenType.UNKNOWN
        elif sp.is_control(i):
            t = gguf.TokenType.CONTROL
        elif sp.is_byte(i):
            t = gguf.TokenType.BYTE
        else:
            t = gguf.TokenType.NORMAL
        types.append(t)
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(sp.bos_id())
    w.add_eos_token_id(sp.eos_id())
    w.add_pad_token_id(sp.pad_id())
    w.add_unk_token_id(sp.unk_id())
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    w.add_add_space_prefix(True)

    def t(name, tensor):
        w.add_tensor(name, tensor.to(torch.float16).numpy())

    t("token_embd.weight", sd["tok_emb.weight"])
    t("output_norm.weight", sd["norm.weight"])
    t("output.weight", sd["tok_emb.weight"])  # tied
    L = cfg["num_layers"]
    for i in range(L):
        p = f"layers.{i}."
        t(f"blk.{i}.attn_norm.weight", sd[p + "attn_norm.weight"])
        t(f"blk.{i}.attn_q.weight", sd[p + "attn.q_proj.weight"])
        t(f"blk.{i}.attn_k.weight", sd[p + "attn.k_proj.weight"])
        t(f"blk.{i}.attn_v.weight", sd[p + "attn.v_proj.weight"])
        t(f"blk.{i}.attn_output.weight", sd[p + "attn.o_proj.weight"])
        t(f"blk.{i}.ffn_norm.weight", sd[p + "ffn_norm.weight"])
        t(f"blk.{i}.ffn_gate.weight", sd[p + "ffn.gate_proj.weight"])
        t(f"blk.{i}.ffn_up.weight", sd[p + "ffn.up_proj.weight"])
        t(f"blk.{i}.ffn_down.weight", sd[p + "ffn.down_proj.weight"])
        if i % 8 == 0:
            print(f"  layer {i}/{L}")

    print("writing gguf ...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"done: {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
