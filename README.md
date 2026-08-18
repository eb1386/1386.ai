# Plasma

A 521M-parameter LLM trained completely from scratch on a single RTX 5080.
No pretrained weights, no cloud, no distillation. 10B tokens of pretraining,
a full SFT pipeline, a custom tokenizer, and a local chat UI.

It seemed dumb. Then an audit found five pipeline bugs that were making it
look about 30% dumber than it actually was.

## The five bugs

**1. The model never saw its own chat template at inference.**
Training tokenized `Assistant: ` as its own span, producing `['▁Assistant', ':']`.
Inference encoded the whole prompt string, which after a newline tokenizes as
`['Ass', 'istant', ':']`. Every generation started from token IDs the model
had never seen in that position. Fixed by building prompts at the token level
(`src/inference/template.py`), exactly mirroring how the SFT data was encoded.

**2. The repetition penalty was banning the answer.**
Penalty 1.3 applied before the temperature divide is roughly a 100x
suppression of repeated tokens at T=0.5. With split-digit tokenization,
answering "85 + 34 = 119" requires repeating digit tokens you just emitted.
Math scored 0.00 with the penalty, 0.62 without it. It also progressively
suppressed "the", "is" and "." which is exactly why answers started correct
and decayed into word salad as they got longer.

**3. The tokenizer could not represent indentation.**
SentencePiece defaults to `remove_extra_whitespaces=True`, which makes
`'    return x'` encode identically to `'return x'`. Ten billion pretraining
tokens contained zero indented code, and the model could never emit valid
Python no matter what. Bonus: the tokenizer's training sample was supposed to
be 10% code, but the code file was empty and the sampler silently
renormalized the weights. The 1.2 tokenizer fixes all of this and verifies
itself before it will accept the trained model.

**4. It was trained not to stop.**
EOS appeared only at the end of whole conversations, so a third of all
answers the model saw were followed by more dialogue instead of a stop token.
Packed training sequences read `answer <EOS> User: new topic` with full
attention across the seam. Under greedy decoding the model stopped at EOS
only 31% of the time. The v4 SFT supervises EOS after every single answer and
uses block-diagonal attention so packed conversations cannot see each other.

**5. The SFT data taught the bad habits.**
UltraChat was 42% of the training signal: verbose essays with fabricated
facts. hh_helpful trained hedge-openers ("I'm not sure, but I think...") at
10-60x the rate of any other source. OASST taught it to claim to be a
different AI assistant, verbatim, at an 83% duplication rate. Under 1% of
conversations contained a Python function.

## Results

Fixing only the serving layer, same checkpoint, zero retraining:

| 309-prompt battery        | before | after |
|---------------------------|--------|-------|
| overall (243 auto-scored) | 0.584  | 0.745 |
| math word problems        | 0.10   | 0.70  |
| code that executes        | 0.00   | 0.20  |
| fake dialogue turns       | 17%    | 0.3%  |

Retraining the SFT on audited data (v4) on top of that:

| behavior                  | v3     | v4    |
|---------------------------|--------|-------|
| stops at EOS              | 54%    | 99.4% |
| mean answer length        | 147 tok| 28 tok|
| instruction following     | 0.33   | 0.53  |
| hedging / identity leaks  | ~1%    | 0.0%  |

Against the previous generation on standard benchmarks (acc_norm, 300/task):
HellaSwag 0.47, PIQA 0.71, ARC-Easy 0.46, all up 8-14 points over the
pre-rebuild model of the same size.

## What it still can't do

Honesty section. Carry arithmetic (26+58 comes out wrong), multi-step
reasoning, and deep factual knowledge past the first sentence. Those are
pretraining-scale limits at 10B tokens, not bugs. Plasma 1.2 (756M, 30B
tokens, lossless tokenizer, seq 2048) targets exactly these and the full
pipeline for it is in this repo.

## Run it

```bash
pip install -r requirements.txt

python run.py                          # chat ui at localhost:8000
python scripts/run_plasma_v3.py        # rebuild 1.1 from scratch (days)
python scripts/run_plasma_1.2.py       # the full 1.2 pipeline (weeks)
python scripts/monitor.py --watch 120  # watch a training run
```

Evaluation:

```bash
python scripts/audit/run_battery.py --subset full --conditions fixed --out logs/audit/my_run.jsonl
python scripts/audit/score_battery.py logs/audit/my_run.jsonl
```

## Architecture

LLaMA-style decoder: 26 layers, hidden 1280, 20 heads with 4 KV heads (GQA),
SwiGLU FFN, RMSNorm, RoPE, tied embeddings, 48k SentencePiece vocab with
split digits and byte fallback, seq 1024. Trained in bf16 with gradient
checkpointing, WSD schedule with (1-sqrt) cooldown, per-source epoch caps so
the data mixture never drifts. Everything fits and trains on one 16GB
consumer GPU.

Weights are not in the repo (GitHub caps files at 100MB; these are 6GB).
Everything needed to reproduce them is.
