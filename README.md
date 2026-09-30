# Plasma

Plasma is a 521M-parameter language model trained from scratch on a single RTX 5080.

It was pretrained on 10B tokens, then instruction-tuned on roughly 300k filtered conversations. The project includes the tokenizer, data pipeline, model architecture, training loop, evaluation tools, inference stack, and local chat UI.

## Model

- 521M parameters
- 26 transformer layers
- Hidden size: 1280
- 20 attention heads
- 4 KV heads using grouped-query attention
- SwiGLU feed-forward layers
- RMSNorm
- RoPE
- Tied input and output embeddings
- 48k SentencePiece vocabulary
- Split-digit tokenization
- Byte fallback
- Context length: 1024

Training used bf16, gradient checkpointing, and a WSD learning-rate schedule with cooldown.

## Training

Plasma was trained completely from scratch.

- Pretraining data: 10B tokens
- Hardware: single RTX 5080 with 16 GB VRAM
- Pretrained weights: none
- Distillation: none
- Cloud compute: none
- Instruction tuning: approximately 300k filtered conversations

The full training pipeline, tokenizer, model code, data preparation, checkpointing, and evaluation scripts are included in this repository.

## Evaluation

Standard benchmark results:

| Benchmark | Score |
|---|---:|
| PIQA | 0.71 |
| HellaSwag | 0.47 |
| ARC-Easy | 0.46 |

These results use `acc_norm` on 300 examples per benchmark.

The repository also includes a 309-prompt behavioral evaluation battery covering instruction following, math, executable code, response termination, formatting, and dialogue behavior.

Selected results after serving and SFT fixes:

| Metric | Result |
|---|---:|
| Overall auto-scored battery | 0.745 |
| Math word problems | 0.70 |
| Executable code | 0.20 |
| Stops at EOS | 99.4% |
| Instruction following | 0.53 |

## Limitations

Plasma is still a small language model trained on a relatively limited token budget.

It remains weak at:

- multi-step reasoning
- arithmetic with carries
- deep factual knowledge
- more difficult coding tasks

These limitations are mainly related to model and pretraining scale.

## Run

```bash
pip install -r requirements.txt
python run.py
