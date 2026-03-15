# 1386.ai

A lightweight transformer language model built from scratch in PyTorch, trained on a single consumer GPU with a full pipeline for data processing, pretraining, and instruction tuning.

No pretrained weights, no HuggingFace model downloads. Every weight is learned from raw text on a single RTX 5080 using bf16 mixed precision with gradient checkpointing. The training infrastructure handles everything from data download through evaluation.

The current release is **Plasma 1.0** (235M parameters). **Plasma 1.1** (500M parameters, multi-turn conversation support, upgraded data pipeline) is in development.

---

## Architecture

The model follows the LLaMA architecture with modern training techniques throughout.

**Attention** uses Grouped-Query Attention (GQA) with query heads mapped to fewer key-value heads, reducing memory bandwidth during inference while maintaining quality. All positional information comes from Rotary Positional Embeddings (RoPE), encoding position directly into the attention computation rather than through learned position embeddings. KV caching is supported for fast autoregressive generation.

**Feed-forward layers** use SwiGLU, a gated activation function that replaces the traditional ReLU MLP. SwiGLU uses three linear projections (gate, up, down) with a SiLU-gated element-wise product, consistently outperforming standard two-projection FFNs at the same parameter count.

**Normalization** is RMSNorm applied before each sub-layer (pre-norm). RMSNorm drops the mean-centering of LayerNorm and only normalizes by the root mean square, which is faster and more stable during mixed-precision training.

The embedding and output projection weights are tied, cutting the parameter count without hurting performance.

### Plasma 1.0

| | |
|---|---|
| Parameters | 235M |
| Hidden size | 1024 |
| Layers | 18 |
| Attention | 16 heads, 4 KV heads (GQA) |
| FFN | SwiGLU, 2816 intermediate |
| Context | 1024 tokens |
| Vocab | 32,000 (SentencePiece BPE) |
| Precision | bf16 |

### Plasma 1.1 (in development)

| | |
|---|---|
| Parameters | 500M |
| Hidden size | 1280 |
| Layers | 26 |
| Attention | 20 heads, 4 KV heads (GQA) |
| FFN | SwiGLU, 3584 intermediate |
| Context | 1024 tokens |
| Vocab | 48,000 (SentencePiece BPE, byte fallback) |
| Multi-turn | yes |

---

## Demo (Plasma 1.0)

> **You:** Hello!
>
> **1386.ai:** Hi there, I'm the one who wrote this. How can I help you today?

> **You:** Name two famous people.
>
> **1386.ai:** Two of the most well-known individuals are John F. Kennedy, who was the 35th President of the United States and George W. Bush, a former president from the same country.

> **You:** When was World War 1?
>
> **1386.ai:** World War I began on June 26, 1914.

> **You:** Name 4 fruits.
>
> **1386.ai:** Fruits are the most common fruit that people eat, and they can be found in a variety of different types such as apples, oranges, bananas, pears, etc

> **You:** What is a steak made of?
>
> **1386.ai:** A steak can be made from various types of meat, including beef.

---

## Training

Training happens in two phases: pretraining on a large filtered corpus, then instruction tuning with loss masking.

**Pretraining** trains the model on billions of tokens of cleaned, deduplicated text from multiple sources. Training uses mixed-precision bf16 with gradient checkpointing to fit on a single consumer GPU. The learning rate follows a cosine schedule with linear warmup.

**Instruction tuning** teaches the model to follow a conversational format. Loss masking ensures the model only learns from assistant response tokens. User prompts are masked during backpropagation. Plasma 1.1 extends this to multi-turn conversations, masking all user turns across the full conversation history.

### Training Plasma 1.1

The 1.1 pipeline is a 12-stage process that handles everything from data download to a final inference test.

| Stage | What it does |
|-------|-------------|
| 0. Cleanup | Free disk from old checkpoints |
| 1. Download | Multi-source: FineWeb-Edu, Wikipedia, StackExchange, code (StarCoder), ArXiv |
| 2. Train classifiers | Train fasttext quality classifier on FineWeb-Edu scores + toxicity classifier on Jigsaw/Civil Comments |
| 3. Quality + toxicity scoring | Classifier-scored quality filtering (60% classifier, 40% heuristics) plus toxic content removal |
| 4. MinHash dedup | Near-duplicate removal across the entire corpus using locality-sensitive hashing |
| 5. Train tokenizer | 48k vocab SentencePiece BPE on 2 GB diverse sample with byte fallback |
| 6. Mix and shard | Domain-weighted mixing (45% web, 15% wiki, 15% code, 10% Q&A, etc.) then tokenization |
| 7. Pretrain | 200k steps, 500M parameters |
| 8. Synthetic instruct | Generate 50k instruction pairs using Claude API (optional) |
| 9. Build instruct shards | Multi-turn loss masking across all instruct sources |
| 10. Finetune | 30k steps with masked loss |
| 11. Test | Inference on benchmark prompts |

Run the full pipeline:

```bash
python scripts/run_1.1.py
```

Run individual stages:

```bash
python scripts/run_1.1.py --stage download
python scripts/run_1.1.py --stage classifiers
python scripts/run_1.1.py --stage quality
python scripts/run_1.1.py --stage dedup
python scripts/run_1.1.py --stage tokenizer
python scripts/run_1.1.py --stage shards
python scripts/run_1.1.py --stage pretrain
python scripts/run_1.1.py --stage synthetic
python scripts/run_1.1.py --stage instruct
python scripts/run_1.1.py --stage finetune
```

To resume pretraining from a checkpoint:

```bash
python -m src.train.train --config configs/pretrain_1.1.yaml --resume checkpoints/1.1_step_50000.pt
```

### Synthetic Instruction Data (optional)

The pipeline can generate high-quality instruction-response pairs using the Claude API. This has the highest impact on instruction following quality for small models.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_synthetic.py --n-samples 50000
```

---

## Infrastructure

- **Data processing** (`src/data/`): quality scoring, MinHash dedup, domain mixing, streaming shard datasets
- **Model** (`src/model/`): transformer with GQA, SwiGLU, RoPE, RMSNorm, KV cache
- **Training** (`src/train/`): gradient accumulation, mixed precision, cosine LR, checkpointing
- **Inference** (`src/inference/`): autoregressive generation with KV caching, temperature/top-k sampling
- **Evaluation** (`src/eval/`): perplexity, math benchmarks, code benchmarks
- **Web UI** (`web/`): FastAPI backend with model management and switching

---

## Running

```bash
pip install -r requirements.txt
python run.py
```

Opens the web UI at `http://localhost:8000`. Available models are detected automatically from the checkpoints directory.

---

MIT License.
