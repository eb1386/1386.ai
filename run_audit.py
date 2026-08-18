import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_checkpoint, load_config
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load actual config from pretrain_1.1.yaml
cfg = load_config("configs/pretrain_1.1.yaml")
config = ModelConfig.from_dict(cfg["model"])
print(f"Model config: vocab={config.vocab_size}, hidden={config.hidden_size}, layers={config.num_layers}")

# ==== TEST 1: Output shape, dtype, range, NaN/Inf check ====
print("\n" + "=" * 70)
print("TEST 1: Forward Pass Validation (1.1_step_210000.pt)")
print("=" * 70)

model = Transformer(config).to(device)
load_checkpoint("checkpoints/1.1_step_210000.pt", model)
model.eval()

batch_size, seq_len = 2, 16
input_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(input_tokens)

print(f"Input shape: {input_tokens.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
print(f"Shape match: {output.shape == (batch_size, seq_len, config.vocab_size)}")
print(f"Output dtype: {output.dtype}")
print(f"Output range: min={output.min().item():.4f}, max={output.max().item():.4f}")
print(f"Output mean: {output.mean().item():.4f}, std={output.std().item():.4f}")
print(f"Has NaN: {torch.isnan(output).any().item()}")
print(f"Has Inf: {torch.isinf(output).any().item()}")

# ==== TEST 2: Weight tying check ====
print("\n" + "=" * 70)
print("TEST 2: Weight Tying (tok_emb vs output)")
print("=" * 70)

emb_weight = model.tok_emb.weight
out_weight = model.output.weight

print(f"Embedding weight shape: {emb_weight.shape}")
print(f"Output weight shape: {out_weight.shape}")
print(f"Same parameter (id check): {id(emb_weight) == id(out_weight)}")
print(f"Tied: {emb_weight is out_weight}")

emb_mean = emb_weight.mean().item()
emb_std = emb_weight.std().item()
out_mean = out_weight.mean().item()
out_std = out_weight.std().item()

print(f"Embedding: mean={emb_mean:.6f}, std={emb_std:.6f}")
print(f"Output:    mean={out_mean:.6f}, std={out_std:.6f}")
print(f"Stats identical: {abs(emb_mean - out_mean) < 1e-6}")

# ==== TEST 3: Activation magnitude per layer ====
print("\n" + "=" * 70)
print("TEST 3: Per-Layer Activation Magnitude (sample 10 layers)")
print("=" * 70)

model.eval()
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        mag = output.abs().mean().item()
        activations[name] = mag
    return hook

handles = []
sample_layers = [0, 6, 12, 18, 24] + list(range(24, min(26, 24+2)))
for i in sample_layers:
    if i < len(model.layers):
        h = model.layers[i].register_forward_hook(hook_fn(f"layer_{i}"))
        handles.append(h)

with torch.no_grad():
    _ = model(input_tokens)

for i in sample_layers:
    if i < len(model.layers):
        mag = activations.get(f"layer_{i}", 0)
        print(f"Layer {i}: activation magnitude = {mag:.6f}")

for h in handles:
    h.remove()

# Check for explosion or collapse
mags = [activations.get(f"layer_{i}", 0) for i in sample_layers if i < len(model.layers)]
if mags:
    print(f"\nMin magnitude: {min(mags):.6f}, Max magnitude: {max(mags):.6f}")
    print(f"Magnitude ratio (max/min): {max(mags) / (min(mags) + 1e-8):.2f}x")
    if max(mags) > 10.0:
        print("WARNING: Activation explosion detected!")
    if min(mags) < 0.01:
        print("WARNING: Activation collapse detected!")

# ==== TEST 4: Gradient check ====
print("\n" + "=" * 70)
print("TEST 4: Gradient Flow Check")
print("=" * 70)

model_train = Transformer(config).to(device)
load_checkpoint("checkpoints/1.1_step_210000.pt", model_train)
model_train.train()

input_tokens = torch.randint(0, config.vocab_size, (2, 16), device=device)
target_tokens = torch.randint(0, config.vocab_size, (2, 16), device=device)

logits = model_train(input_tokens)
loss = nn.CrossEntropyLoss()(logits.view(-1, config.vocab_size), target_tokens.view(-1))
loss.backward()

grad_norms = {}
zero_grad_layers = []
for name, param in model_train.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms[name] = grad_norm
        if grad_norm == 0:
            zero_grad_layers.append(name)

print(f"Total params with gradients: {len(grad_norms)}")
print(f"Params with zero gradients: {len(zero_grad_layers)}")

# Sample layer gradient norms
layer_grads = {}
for name, norm in grad_norms.items():
    for i in range(len(model_train.layers)):
        if f"layers.{i}" in name:
            layer_grads.setdefault(f"layer_{i}", []).append(norm)
            break

print("\nGradient norms (early vs late):")
for i in [0, 6, 12, 18, 24]:
    norms = layer_grads.get(f"layer_{i}", [])
    if norms:
        mean_norm = sum(norms) / len(norms)
        print(f"Layer {i}: mean grad norm = {mean_norm:.6f}")

# ==== TEST 5: Generation test ====
print("\n" + "=" * 70)
print("TEST 5: Generation Test (greedy decoding)")
print("=" * 70)

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(cfg["data"]["tokenizer_path"])

model.eval()
prompt = "The capital of Australia is"
tokens = tokenizer.encode(prompt, out_type=int)

print(f"Prompt: '{prompt}'")

# Greedy decode 10 tokens
with torch.no_grad():
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    kv_caches = [None] * len(model.layers)
    
    logits, kv_caches = model(x, start_pos=0, kv_caches=kv_caches)
    next_logits = logits[0, -1, :]
    
    generated = []
    for _ in range(10):
        next_token = next_logits.argmax().item()
        generated.append(next_token)
        
        x = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, kv_caches = model(x, start_pos=len(tokens) + len(generated) - 1, kv_caches=kv_caches)
        next_logits = logits[0, -1, :]

generated_text = tokenizer.decode(generated)
print(f"Generated (210k): '{generated_text}'")

# ==== TEST 6: Compare 170k vs 210k ====
print("\n" + "=" * 70)
print("TEST 6: Comparison 1.1_step_170000.pt vs 1.1_step_210000.pt")
print("=" * 70)

model_170k = Transformer(config).to(device)
load_checkpoint("checkpoints/1.1_step_170000.pt", model_170k)
model_170k.eval()

with torch.no_grad():
    tokens_170k = tokenizer.encode(prompt, out_type=int)
    x = torch.tensor([tokens_170k], dtype=torch.long, device=device)
    kv_caches_170k = [None] * len(model_170k.layers)
    
    logits_170k, _ = model_170k(x, start_pos=0, kv_caches=kv_caches_170k)
    next_logits_170k = logits_170k[0, -1, :]
    
    generated_170k = []
    for _ in range(10):
        next_token = next_logits_170k.argmax().item()
        generated_170k.append(next_token)
        
        x = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits_170k, _ = model_170k(x, start_pos=len(tokens_170k) + len(generated_170k) - 1, 
                                   kv_caches=kv_caches_170k)
        next_logits_170k = logits_170k[0, -1, :]

generated_text_170k = tokenizer.decode(generated_170k)
print(f"Generated (170k): '{generated_text_170k}'")
print(f"Generated (210k): '{generated_text}'")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
