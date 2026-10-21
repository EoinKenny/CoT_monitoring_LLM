# Qwen3.5-122B-A10B-FP8 backbone collection

Authorized 7 October 2026: after a successful smoke test and sufficient disk space,
collect the full backbone data automatically. This controller collects features;
it does not submit judge calls or change the existing experiments.

## Fixed scientific protocol

- Official model: `Qwen/Qwen3.5-122B-A10B-FP8`, revision
  `a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9`.
- Extract the shared residual stream after block index 31 (32nd of 48), before
  final normalization. Retain all experts in each executed block. Later blocks,
  vision components and the language-model head cannot affect this activation
  and are omitted. A small native 48-block architectural test compares the full
  model's block-31 hook with the truncated prefix, requiring bitwise equality.
- This is a 122B-total / approximately 10B-active MoE in official FP8 precision,
  **not** a dense 122B model or an unquantized scaling comparison.
- Reuse all 456,008 frozen eligible records in the ten existing datasets. Preserve
  their immutable IDs, original messages, labels, train/validation/test membership,
  source groups and input hashes. Assert no cross-split group/input-hash overlap.
- Render the pinned native tokenizer/chat template, no generation prompt, thinking
  disabled, fixed date argument. Never truncate. Locate content by sentinel
  substitution and exact rendered-text reconstruction. Exclude special tokens and
  padding. Pool the same role policy as each original record (`user`, `assistant`,
  or `all`); save mean-pooled content activations in float32, dimension 3072.
- Explicit template adaptation: Qwen3.5 rejects conversations without a user turn.
  For those source records only, insert an empty user turn after a leading system
  message (otherwise at the beginning). It adds no content to the pooling mask.
  Original messages remain stored unchanged; adapted rows are marked
  `empty_user_template_adapter=true`. Earlier compatible wildguard preparation
  predates that boolean field; no rows there required the adapter. Any other
  nonliteral template transformation stops preparation for explicit review.
- Frozen batches: sort by native token length then immutable ID, max four rows and
  4096 padded tokens; longer records alone. Preserve original row indices in the
  final matrix and save the exact batch schedule. No truncation or data balancing.

## Execution and verification

Remote smoke/code directory: `/home/ubuntu/qwen122b-fp8-smoke-20261007`.
Output directory: `/home/ubuntu/qwen122b-fp8-full-20261007`.
Model files: `/opt/dlami/nvme/qwen122b-fp8-smoke-model/<revision>`.
All are separate from the experiments currently being migrated.

`download.py` saves a pinned download plan and per-file SHA-256 proofs, checking
published LFS hashes where available. It downloads only shards containing needed
prefix tensors, although shard layout means 37 of 39 weight shards are required.
`collect.py` verifies complete source-key and target-parameter coverage, combining
each expert's gate/up weights and scales into the native Transformers layout.
Checkpoint weight/state dtypes are preserved. BF16/FP16 scaling factors are
promoted exactly to FP32 (no value change) for the runtime's scale layout; the
loader records how many source tensors required this promotion. An initial
strict dtype assertion caught the BF16 scales before any activations were saved;
that failure and the reviewed restart are retained.

Runtime uses the new isolated `env/bin/python` (Python 3.12), with the migrated
vLLM environment's site-packages exposed read-only through `PYTHONPATH`.
`kernels==0.17.2` is installed only in the isolated environment. The official
finegrained-fp8 v4 kernel sources are copied to `frozen_fp8_kernel`, checksummed
in `KERNEL_MANIFEST.json`, and loaded locally to avoid mutable Hub refs. The
Transformers grouped FP8 experts implementation is used. DeepGEMM is disabled;
SDPA uses its math backend. Package versions are saved in `packages.txt`.

Seed: 20261004. Python hash seed is fixed at launch. Torch deterministic algorithms
are required (errors are not downgraded); TF32 and cuDNN benchmarking are disabled.
GPU architecture and Torch/CUDA versions are recorded. Triton kernel compilation
and autotuning, FP8 numerics, hardware and package differences mean that fresh
cross-machine numerical reproduction is distinct from exact saved-artifact replay.
The saved feature matrices and checksummed commits are authoritative.

Smoke acceptance requires finite, nonzero pooled vectors for all ten datasets,
successful longest-record execution, bitwise repeatability for identical single
and padded batches, and cosine similarity above 0.999 for the same short records
executed singly versus padded together. A four-record, approximately 4096-token
batch tests the intended batch budget. The kernel preflight also compares grouped
FP8 expert output against the eager reference implementation and checks exact
repeatability. Test records use existing frozen IDs, not newly sampled datasets.

Collection starts automatically only after `SMOKE_PASSED.json` is saved and the
disk reserve check passes. It reserves at least 20 GiB of main-disk free space,
plus the next output allocation. Canonical feature matrices need about 5.22 GiB;
metadata, logs and commit proofs require additional space.

Each commit covers up to 25 fixed batches, recording IDs, original row indices,
feature checksum, metadata/schedule hashes and timing/memory measurements. Flush
and fsync precede the atomic commit. On replay, completed commits are verified.
An uncommitted `.pending.json` causes a stop for review rather than silent
regeneration. Failure records and logs are retained. Completed datasets have
whole-file checksums; `COMPLETED.json` confirms all 456,008 rows.

## Status and launch

Read output `STATUS.json`, smoke `COLLECTION_PID`, the corresponding collection
log, GPU/process state and filesystem space together. A running controller may
still be preparing/tokenizing/loading/testing; it is not evidence that activation
collection has started. `SMOKE_PASSED.json` and `state=extracting` establish that.

Launch detached with the isolated Python and these environment variables:

```
PYTHONPATH=/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages
HF_HUB_OFFLINE=1
PYTHONHASHSEED=20261004
```

Preserve the previous log and diagnostic files on any reviewed restart. A file
lock prevents simultaneous collectors. The first two preparation attempts stopped
on the native template's missing-user requirement; no activation rows were
collected. The explicit empty-user adapter addresses that failure.

The authorized obsolete GPTQ cache deletion is separately documented in
`GPTQ_DELETION_MANIFEST.json` and `GPTQ_DELETED.json`. Only about 39 MiB remained;
the large GPTQ weights were already absent. Existing runs, environments, current
weights and GPU keepalives were preserved.
