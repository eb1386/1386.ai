import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.train.utils import load_checkpoint, load_config
import sentencepiece as spm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config("configs/pretrain_1.1.yaml")
config = ModelConfig.from_dict(cfg["model"])

print("=" * 70)
print("ADDITIONAL HEALTH CHECKS: Extended Gradient and Layer Analysis")
print("=" * 70)

# Load model
model = Transformer(config).to(device)
load_checkpoint("checkpoints/1.1_step_210000.pt", model)
model.train()

# Create batch and forward
input_tokens = torch.randint(0, config.vocab_size, (2, 32), device=device)
target_tokens = torch.randint(0, config.vocab_size, (2, 32), device=device)

logits = model(input_tokens)
loss = nn.CrossEntropyLoss()(logits.view(-1, config.vocab_size), target_tokens.view(-1))
loss.backward()

print(f"\nLoss value: {loss.item():.6f}")

# Comprehensive gradient stats across ALL layers
layer_grad_data = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        for i in range(len(model.layers)):
            if f"layers.{i}" in name:
                if i not in layer_grad_data:
                    layer_grad_data[i] = []
                layer_grad_data[i].append(grad_norm)
                break

print("\nGradient Statistics Across All Layers:")
print("(Early=start, Late=end)")
early_norms = []
late_norms = []
for i in range(len(model.layers)):
    norms = layer_grad_data.get(i, [])
    if norms:
        mean_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        min_norm = min(norms)
        if i < 8:
            early_norms.append(mean_norm)
        if i >= 18:
            late_norms.append(mean_norm)
        print(f"Layer {i:2d}: mean={mean_norm:.6f}, max={max_norm:.6f}, min={min_norm:.6f}")

if early_norms and late_norms:
    early_avg = sum(early_norms) / len(early_norms)
    late_avg = sum(late_norms) / len(late_norms)
    ratio = late_avg / (early_avg + 1e-8)
    print(f"\nEarly layers avg norm: {early_avg:.6f}")
    print(f"Late layers avg norm: {late_avg:.6f}")
    print(f"Gradient diminishing ratio (late/early): {ratio:.4f}")
    if ratio < 0.1:
        print("WARNING: Severe gradient flow problem detected!")
    elif ratio < 0.5:
        print("WARNING: Moderate gradient diminishing detected")

# Check parameter statistics
print("\n" + "=" * 70)
print("Parameter Statistics")
print("=" * 70)

param_stats = {}
for name, param in model.named_parameters():
    if param.requires_grad:
        p_mean = param.data.mean().item()
        p_std = param.data.std().item()
        p_norm = param.data.norm().item()
        
        for i in range(len(model.layers)):
            if f"layers.{i}" in name:
                if i not in param_stats:
                    param_stats[i] = []
                param_stats[i].append((name.split(".")[-1], p_mean, p_std, p_norm))
                break

# Show norms for sampled layers
for i in [0, 6, 12, 18, 24, 25]:
    if i in param_stats:
        params = param_stats[i]
        print(f"\nLayer {i}:")
        for pname, pmean, pstd, pnorm in params[:3]:
            print(f"  {pname}: norm={pnorm:.4f}, mean={pmean:.6f}, std={pstd:.6f}")

# Final comprehensive logits check
print("\n" + "=" * 70)
print("Logits Health Check (on eval)")
print("=" * 70)

model.eval()
with torch.no_grad():
    test_input = torch.randint(0, config.vocab_size, (4, 64), device=device)
    logits = model(test_input)
    
    # Check logit statistics
    logits_flat = logits.view(-1)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {logits_flat.mean().item():.6f}")
    print(f"Logits std: {logits_flat.std().item():.6f}")
    print(f"Logits max: {logits_flat.max().item():.6f}")
    print(f"Logits min: {logits_flat.min().item():.6f}")
    print(f"Logits median: {logits_flat.median().item():.6f}")
    
    # Check softmax
    probs = torch.softmax(logits, dim=-1)
    print(f"\nProb max (should be near 1): {probs.max().item():.6f}")
    print(f"Prob sum (should be 1): {probs.sum(dim=-1).mean().item():.6f}")
    print(f"Top-1 accuracy on random targets: {(logits.argmax(-1) == torch.randint(0, config.vocab_size, logits.shape[:-1], device=device)).float().mean().item() * 100:.2f}%")

print("\n" + "=" * 70)
print("EXTENDED AUDIT COMPLETE")
print("=" * 70)
