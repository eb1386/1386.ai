#!/usr/bin/env python3
# wait for synthetic data then train

import time
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = ROOT / "data" / "raw_1.1" / "synthetic_instruct.jsonl"
TARGET = 15000

print("Watching synthetic data generation...")
print(f"  Target: {TARGET:,} samples")
print(f"  Checking every 30s\n")

last_count = 0
stall_checks = 0

while True:
    if SYNTHETIC.exists():
        count = sum(1 for _ in open(SYNTHETIC, encoding="utf-8"))
        print(f"  {count:,}/{TARGET:,} samples", flush=True)

        if count >= TARGET:
            print(f"\nTarget reached ({count:,} samples). Starting pipeline.\n")
            break

        # detect if generation stopped
        if count == last_count:
            stall_checks += 1
        else:
            stall_checks = 0
        last_count = count

        # if stalled for 5 min and we have enough data, start anyway
        if stall_checks >= 10 and count >= 5000:
            print(f"\nGeneration stalled at {count:,} samples. Starting pipeline anyway.\n")
            break
    time.sleep(30)

# run full pipeline
subprocess.run([sys.executable, "-u", "scripts/run_1.1.py"], cwd=str(ROOT))
