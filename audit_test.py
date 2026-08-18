import sys
sys.path.insert(0, '.')
from src.data.dataset import StreamingShardDataset
import numpy as np
from pathlib import Path

# Test 1: Verify worker distribution
print("=== TEST 1: Worker Shard Distribution ===")
dataset = StreamingShardDataset("data/shards_1.1", split="train")
print(f"Total shards loaded: {len(dataset.pairs)}")

# Simulate worker_info
class FakeWorkerInfo:
    def __init__(self, id, num):
        self.id = id
        self.num_workers = num

num_shards = len(dataset.pairs)
print(f"Train shards: {num_shards}")

for num_workers in [1, 2, 4, 8]:
    print(f"\nWith num_workers={num_workers}:")
    total_coverage = 0
    for worker_id in range(num_workers):
        my_pairs = dataset.pairs[worker_id::num_workers]
        total_coverage += len(my_pairs)
        print(f"  Worker {worker_id}: {len(my_pairs)} shards")
    if total_coverage != num_shards:
        print(f"  WARNING: Total coverage = {total_coverage} != {num_shards}!")

# Test 2: Verify seed independence
print("\n=== TEST 2: Seed Independence Per Worker ===")
seed = 42
for worker_id in range(2):
    rng = np.random.default_rng(seed + worker_id * 1000003)
    order1 = list(range(5))
    rng.shuffle(order1)
    
    rng2 = np.random.default_rng(seed + worker_id * 1000003)
    order2 = list(range(5))
    rng2.shuffle(order2)
    
    print(f"Worker {worker_id} shuffle 1: {order1}")
    print(f"Worker {worker_id} shuffle 2: {order2}")
    if order1 != order2:
        print("  ERROR: Same seed + worker_id gives different shuffles!")

# Test 3: Duplicate check at generator level
print("\n=== TEST 3: Check For Shard Duplication ===")
my_pairs_w0 = dataset.pairs[0::2]
my_pairs_w1 = dataset.pairs[1::2]
print(f"Worker 0 shards: {[p[0].name for p in my_pairs_w0[:3]]}...")
print(f"Worker 1 shards: {[p[0].name for p in my_pairs_w1[:3]]}...")

overlap = set([p[0] for p in my_pairs_w0]) & set([p[0] for p in my_pairs_w1])
if overlap:
    print(f"CRITICAL: {len(overlap)} shards assigned to BOTH workers!")
else:
    print("OK: No shard overlap between workers")

print("\n=== TEST 4: Resume State Bug ===")
print(f"Seed passed to __init__: {dataset.seed}")
print("Each worker adds worker_id * 1000003 to seed.")
print("On resume with same seed, same worker_id gets same shuffle.")
print("If dataset is shuffled but not checkpointed, model sees identical batches twice!")

