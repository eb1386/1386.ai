# text generation sampling

import argparse

import torch
import torch.nn.functional as F
import sentencepiece as spm

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_config, load_checkpoint


def apply_repetition_penalty(
    logits: torch.Tensor,
    past_tokens: list[int],
    penalty: float = 1.2,
    window: int = 256,
    exclude: set[int] | None = None,
) -> torch.Tensor:
    if penalty == 1.0 or not past_tokens:
        return logits
    token_set = set(past_tokens[-window:])
    # punctuation and structure tokens must stay repeatable, or sentences
    # can never end and generation drifts into the weird tail
    if exclude:
        token_set -= exclude
    for token_id in token_set:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    if temperature <= 0:
        return logits.argmax().item()

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs - sorted_probs >= top_p
        sorted_logits[mask] = float("-inf")
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: torch.device = torch.device("cpu"),
    return_new_only: bool = False,
    return_meta: bool = False,
    prompt_ids: list[int] | None = None,
    penalty_window: int = 256,
    penalty_exclude: set[int] | None = None,
    stop_id_seqs: list[list[int]] | None = None,
    no_repeat_ngram: int = 0,
) -> str:
    model.eval()
    # prompt_ids lets the caller build the prompt at token level, matching
    # how sft data was span-encoded; a raw string cannot reproduce those ids
    tokens = list(prompt_ids) if prompt_ids is not None else tokenizer.encode(prompt, out_type=int)
    max_seq = model.config.max_seq_len

    # keep room to actually generate, not just to read the prompt
    keep = max(1, max_seq - max_tokens)
    if len(tokens) > keep:
        tokens = tokens[-keep:]

    # only penalize generated tokens, not prompt
    generated_ids: list[int] = []

    # prefill: process entire prompt, collect kv cache
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    kv_caches = [None] * len(model.layers)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, kv_caches = model(x, start_pos=0, kv_caches=kv_caches)

    def hit_stop_seq():
        # trained continuation like "\n" + "▁User" means a fake turn is starting
        if not stop_id_seqs:
            return False
        for seq in stop_id_seqs:
            n = len(seq)
            if n and len(generated_ids) >= n and generated_ids[-n:] == seq:
                del generated_ids[-n:]
                del tokens[-n:]
                return True
        return False

    def ban_repeated_ngrams(logits):
        # a candidate completing an already-seen n-gram gets banned; kills
        # degenerate loops ("very, very, ..." / "Giraffe" x10) at the source
        n = no_repeat_ngram
        if not n or len(generated_ids) < n:
            return logits
        prefix = tuple(generated_ids[-(n - 1):]) if n > 1 else ()
        for i in range(len(generated_ids) - n + 1):
            if tuple(generated_ids[i:i + n - 1]) == prefix:
                logits[generated_ids[i + n - 1]] = float("-inf")
        return logits

    # fp32 sampling math
    next_logits = logits[0, -1, :].float()
    next_logits = apply_repetition_penalty(next_logits, generated_ids, repetition_penalty,
                                           penalty_window, penalty_exclude)
    next_token = sample_top_k_top_p(next_logits, temperature, top_k, top_p)

    # why generation ended, for audits
    stop_reason = "eos"
    if next_token != tokenizer.eos_id():
        stop_reason = "length"
        tokens.append(next_token)
        generated_ids.append(next_token)
        cur_pos = len(tokens) - 1  # position of the token we just appended

        # decode: one token at a time using cached keys/values
        for _ in range(max_tokens - 1):
            if hit_stop_seq():
                stop_reason = "stop_seq"
                break
            if cur_pos >= max_seq:
                stop_reason = "context"
                break

            x = torch.tensor([[next_token]], dtype=torch.long, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, kv_caches = model(x, start_pos=cur_pos, kv_caches=kv_caches)

            next_logits = logits[0, -1, :].float()
            next_logits = apply_repetition_penalty(next_logits, generated_ids, repetition_penalty,
                                                   penalty_window, penalty_exclude)
            next_logits = ban_repeated_ngrams(next_logits)
            next_token = sample_top_k_top_p(next_logits, temperature, top_k, top_p)

            if next_token == tokenizer.eos_id():
                stop_reason = "eos"
                break

            tokens.append(next_token)
            generated_ids.append(next_token)
            cur_pos += 1

    text = tokenizer.decode(generated_ids if return_new_only else tokens)
    if return_meta:
        return text, {"stop": stop_reason, "n_new": len(generated_ids)}
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate text")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = ModelConfig.from_dict(cfg["model"])
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(data_cfg["tokenizer_path"])

    print(f"Prompt: {args.prompt}\n")
    print("--- Generated text ---")
    output = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print(output)
    print("--- End ---")


if __name__ == "__main__":
    main()
