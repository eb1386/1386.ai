# Plasma 1.2 — design record

756M dense, 30B tokens, seq 2048, single RTX 5080. Every decision below
traces to the August 2026 audit of 1.1 (see `scripts/audit/` and the audit
report artifact). One command runs everything:

    python scripts/run_plasma_1.2.py            # full pipeline, resume-safe
    python scripts/run_plasma_1.2.py --status   # stage states
    python scripts/run_plasma_1.2.py --until 4  # data prep only (no GPU)

## Architecture

| | 1.1 | 1.2 | why |
|---|---|---|---|
| params | 521M | 756M (measured 756,442,112) | user target ~750M |
| shape | h1280 L26 | h1536 L28, heads 24, kv 4, ffn 4096 | best aspect ratio of 3 audited candidates |
| seq | 1024 | 2048 | whole documents; costs ~10% throughput |
| rope theta | 10k | 100k | standard at 2k ctx, extension headroom |
| qk-norm | — | yes | attention-logit stability over ~1M steps (OLMo-2/Gemma-2) |
| z-loss | — | 1e-4 | logit drift guard on long runs |
| optimizer | adamw fp32 | adamw 8-bit | fp32 states alone are 5.6 GB at 756M; 8-bit fits 16 GB with mb2 headroom |

**Dense over MoE.** A from-scratch PyTorch MoE without grouped-GEMM kernels is
memory-bandwidth-bound on one consumer GPU and lands at or below dense tok/s;
expert weights multiply fixed VRAM (a 1.2B-total MoE ≈ 11.2 GB before
activations); router collapse and capacity tuning are live engineering risk in
a bespoke codebase. Falsification path if MoE tempts again for 1.3: one
top-2-of-8 SwiGLU MoE block benchmarked at matched ACTIVE params via
`scripts/ablate.py` — commit only if it sustains ≥0.9× dense tok/s with stable
routing entropy.

**Learning rate.** Peak 5.0e-4 (1/width transfer from 1.1's comfortable 6e-4
at h1280), warmup 2000 steps, WSD stable to 90%, 1-sqrt cooldown over the last
10% on the anneal mix. 32,768 tokens/step (mb2 × 2048 × accum 8) → 915k steps.

**Honest wall-clock: 28–36 days.** Scaling measured 1.1 throughput by the
FLOPs ratio: ~9.6k tok/s contended / ~12.2k quiet (compiled) at seq 2048.
Options: seq 1024 saves ~10%; WSD means an early cooldown at ANY point yields
a usable model — 20B tokens ≈ 3 weeks forfeits only the tail.

## Tokenizer (retrained — 1.1's was disqualified)

1.1 defects: `remove_extra_whitespaces=True` made indentation UNREPRESENTABLE
(the model literally could not emit valid Python); no newline piece (every \n
fragmented the next word — the cause of the train/serve template seam); the
training sample contained zero code (empty file silently renormalized away).

1.2: lossless whitespace, `\n`/`\t` first-class, atomic `<|user|>`
`<|assistant|>` `<|end|>` `<|doc|>` markers, split digits, byte fallback,
code REQUIRED in the sample with loud failure. `train_tokenizer_1.2.py`
self-verifies round-trips before accepting the model.

## Pretraining data — 30B tokens

Measured per-epoch yields anchored to the real 1.1 shards; nothing exceeds
~1.7 epochs. ~73 GB of new downloads (script: `download_1.2_data.py`,
resumable, `<|doc|>` sentinels so documents survive whole).

| source | share | ~tokens | epochs | note |
|---|---|---|---|---|
| fineweb-edu | 26% | 7.8B | ~1 | fresh 100BT offset past 1.1's pull |
| code (starcoder-py + python-edu) | 17% | 5.1B | ~1.3 | whole files, indentation intact |
| wikipedia FULL | 14% | 4.2B | ~0.9 | 1.1 had only 30% of the dump |
| fineweb general | 10% | 3.0B | ~0.5 | the entity/pop-culture fix, kept |
| books (gutenberg) | 10% | 3.0B | ~0.9 | 1.1 had zero book data |
| finemath 4+/3+ | 9% | 2.7B | ~1.2 | 3plus deduped against 4plus at download |
| cosmopedia v2 | 7% | 2.1B | ~0.7 | offset past 1.1's pull |
| stackexchange | 5% | 1.5B | ~1.7 | |
| arxiv | 2% | 0.6B | ~1.7 | |

Corpus builder scrubs mojibake, exact-hash dedups across web sources (a
boilerplate page appeared 224× in 1.1's corpus), and refuses to start if any
mix path is empty. Anneal mix (~3B: wiki/cosmopedia/code-edu/math/books
heavy) for the cooldown phase.

## SFT

Same audited recipe as 1.1-v4 (`build_sft_v4.py --template special`):
hh_helpful removed (trained hedging), ultrachat 80k→32k + 3-turn cap (was
41.6% of loss tokens), 50k code conversations added (v3 had ~0.8%), identity
scrubbed + synthetic Plasma identity, refusal/hedge openings dropped,
cross-source dedup, `<|end|>` supervised after EVERY assistant turn, BOS doc
boundaries + block-diagonal attention (`doc_attention_bos_id`), decontaminated
against the benchmark and the 309-prompt audit battery.

## Evaluation gates

- `scripts/audit/run_battery.py` — 309-prompt battery, reusable regression
  suite with auto-scoring (`score_battery.py`)
- `scripts/benchmark_v2.py` + `scripts/eval_standard.py` — the 1.1 suites,
  run against 1.1-v3/v4 baselines
- Gate before believing anything: same prompts, same scorer, all models in
  one run.
