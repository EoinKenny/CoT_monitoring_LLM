# One-layer activation timing pilot

This is a bounded timing and storage pilot, not full activation collection or probe training.

The script runs on the existing L40S instance in `/home/ubuntu/experiments/activation-pilot-20261001`. It preserves existing workloads. Model and dataset commit hashes are pinned in `pilot.py` and recorded in the output.

- Model: Llama-3.2-1B-Instruct, BF16, Transformers SDPA, no compilation.
- Saved representation: residual output of transformer block index 10 (the 11th block), all non-padding tokens. The full 16-block backbone runs, without the vocabulary head, KV cache, or generation.
- Inputs: model chat template, no added system instruction or generation prefix. WildGuard uses prompt and response; ToxicChat uses user input only; high-stakes uses the supplied conversation or wraps plain input as a user message. Labels and annotation explanations are never model input.
- Data inventory: eligible response-harm rows from WildGuard train/test; high-stakes `training/` files and all balanced evaluation files; ToxicChat January 2024 train/test. No cross-split deduplication in this timing inventory. Exact files and counts are recorded.
- Sampling: tokenize the text inventory, divide each dataset into eight equal-count length groups, then sample eight examples per group with seed 20261001. Only these 192 examples undergo activation extraction. Batches use eight examples unless a long batch needs splitting to stay within 32,768 padded tokens. No text is truncated.
- Timing: two warmup forwards, then synchronize GPU timing boundaries. Measure input preparation, full backbone forward, unpadding/CPU transfer, and safetensors serialization plus `fsync` separately. File read-back and exact BF16 tensor equality checks are excluded from extraction timing.
- Projection: extrapolate measured seconds per example within each length group to that group's population. Also report a total-token throughput projection as a sensitivity check. Projected times are estimates from a small sample, not measured complete-run times.
- Disk: count all tokens in the chosen inputs. Raw one-layer storage is tokens × 2,048 × 2 bytes; token IDs and file metadata add a small amount. Files are concatenated unpadded tensors with offsets and token IDs.

Outputs include `inventory.json`, `journal.jsonl`, `results.json`, the sample activation files, and a `COMPLETED` marker only after every saved activation tensor passes a read-back check. Restarting the script would repeat the pilot; inspect existing output before running it again.
