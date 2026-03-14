# 1386.ai

A transformer language model trained from scratch in PyTorch. No pretrained weights, no HuggingFace, no shortcuts. Every weight in this model was learned from raw text on a single RTX 5080 — 5 billion tokens, 100k pretraining steps at batch size 32, then 20k instruction tuning steps with loss masking, all in bf16 mixed precision.

The current release is **Plasma 1.0**, a 235M parameter model. **Plasma 1.1** (500M parameters, multi-turn conversation support) is currently in development.

---

## Architecture

The model follows the LLaMA architecture with modern training techniques throughout.

**Attention** uses Grouped-Query Attention (GQA) with 16 query heads mapped to 4 key-value heads, reducing memory bandwidth during inference while maintaining quality. All positional information comes from Rotary Positional Embeddings (RoPE), which encode position directly into the attention computation rather than through learned position embeddings — this gives better length generalization and eliminates the need for absolute position tokens.

**Feed-forward layers** use SwiGLU, a gated activation function that replaces the traditional ReLU-based MLP. SwiGLU uses three linear projections (gate, up, down) with a SiLU-gated element-wise product, which consistently outperforms standard two-projection FFNs at the same parameter count.

**Normalization** is RMSNorm applied before each sub-layer (pre-norm). RMSNorm drops the mean-centering of LayerNorm and only normalizes by the root mean square, which is faster and more numerically stable during mixed-precision training.

The embedding and output projection weights are tied, which cuts the parameter count without hurting performance.

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

### Plasma 1.1 (coming soon)

| | |
|---|---|
| Parameters | 500M |
| Hidden size | 1280 |
| Layers | 26 |
| Attention | 20 heads, 4 KV heads (GQA) |
| FFN | SwiGLU, 3584 intermediate |
| Context | 1024 tokens |
| Multi-turn | yes |

---

## Training

Training happens in two phases: pretraining on a large filtered corpus, then instruction tuning with loss masking.

**Pretraining** trains the model on billions of tokens of cleaned, deduplicated text. Data quality is enforced through paragraph-level deduplication using MD5 hashing, length and repetition filtering, and quality scoring. The training pipeline handles downloading, cleaning, tokenization, and shard packing automatically. Training uses mixed-precision bf16 with gradient checkpointing to fit on a single consumer GPU. The learning rate follows a cosine schedule with linear warmup.

**Instruction tuning** teaches the model to follow a conversational format. Loss masking ensures the model only learns from assistant response tokens — user prompts are masked during backpropagation, which dramatically improves instruction following quality. Plasma 1.1 extends this to multi-turn conversations, masking all user turns across the full conversation history.

The tokenizer is a 32,000-vocab SentencePiece BPE model trained on the same corpus.

### Training Plasma 1.1

Training Plasma 1.1 from scratch takes several days on a consumer GPU. The pipeline downloads data, builds shards, pretrains 200k steps, then finetunes 30k steps.

```bash
python scripts/run_v6_upgrade.py
```

To resume from a checkpoint:

```bash
python -m src.train.train --config configs/pretrain_1.1.yaml --resume checkpoints/1.1_step_50000.pt
```

---

## Running

```bash
pip install torch sentencepiece pyyaml fastapi uvicorn
```

```bash
python run.py
```

Opens the chat interface at `http://localhost:8000`.

---

Constant updates coming as I continue to improve the model. MIT License.
