import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.transformer import Transformer

print("="*60)
print("TEST 1: Weight Tying")
print("="*60)

config = ModelConfig(vocab_size=32000, hidden_size=512, num_layers=2)
model = Transformer(config)

is_tied = model.output.weight is model.tok_emb.weight
print(f"Weight tying: {is_tied}")

print("\n" + "="*60)
print("TEST 2: Parameter Count")
print("="*60)

param_count = model.count_parameters()
estimated = config.param_count_estimate()
print(f"Actual: {param_count:,}")
print(f"Estimated: {estimated:,}")
print(f"Difference: {param_count - estimated:,}")

print("\n" + "="*60)
print("TEST 3: Forward Pass")
print("="*60)

tokens = torch.randint(0, 32000, (2, 128))
logits = model(tokens)
print(f"Output shape: {logits.shape}")
print(f"Expected: torch.Size([2, 128, 32000])")

print("\n" + "="*60)
print("TEST 4: Gradient Flow")
print("="*60)

model.zero_grad()
logits = model(tokens)
loss = logits.sum()
loss.backward()

no_grad = [n for n, p in model.named_parameters() if p.grad is None]
print(f"Parameters with no gradients: {no_grad if no_grad else 'None'}")

print("\n" + "="*60)
print("TEST 5: KV Cache")
print("="*60)

try:
    tokens_single = torch.randint(0, 32000, (1, 1))
    kv_caches = [None] * config.num_layers
    logits, new_caches = model(tokens_single, start_pos=0, kv_caches=kv_caches)
    print(f"Cache logits shape: {logits.shape}")
    print(f"Cache count: {len(new_caches)}")
    print("KV cache test: OK")
except Exception as e:
    print(f"KV cache test: FAILED - {e}")

print("\n" + "="*60)
print("TEST 6: Freqs_cis Persistence")
print("="*60)

in_state = 'freqs_cis' in model.state_dict()
print(f"freqs_cis in state_dict: {in_state}")
if not in_state:
    print("WARNING: freqs_cis NOT in checkpoint!")

