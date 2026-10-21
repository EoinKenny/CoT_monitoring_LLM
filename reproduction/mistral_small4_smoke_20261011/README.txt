Mistral Small 4 feasibility smoke test, 2026-10-11

Official checkpoint: mistralai/Mistral-Small-4-119B-2603-NVFP4
Pinned revision: 45331841b631f4e281df8e959ea3cc9beb84298a
Remote directory: /home/ubuntu/experiments/mistral-small4-smoke-20261011
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB.

Scope: exactly two batches of 16 calls. The same 16 underlying cases are
used for hard-label and probability-only output. Selection covers all ten
datasets, five backbones, and three explanations, with eight positive and
eight negative cases. Each case has one assigned explanation. This is a
feasibility and approximate throughput test, not an explanation ablation or
a reliable accuracy estimate. No full study is started by these scripts.

Existing task messages are taken verbatim from the checksummed reasoning
pilot case artifact; reasoning is disabled using Mistral's official
reasoning_effort='none' template setting. The Mistral release's generic
SYSTEM_PROMPT.txt is archived as part of the checkpoint but does not replace
our frozen study task instructions. No target gold labels enter prompts.

Hard labels: one generated token, constrained to the tokens for 0 and 1.
Probability: integer 0-100 constrained by xgrammar, max_tokens=8.
Both use temperature=0, seed=20261001, batch size 16, eager execution and
no prefix caching. All exact token IDs, raw outputs, versions and settings
are saved. Invalid/ambiguous judge attempts are preserved, never retried
automatically. Runtime extrapolations include first-use overhead and are
only preliminary. Storage guards preserve a 20 GiB reserve.

Transport repair: the initial Hugging Face Python client stalled before
transferring files. A bounded independent diagnostic located the stall in
socket.create_connection, while curl worked. Only that download process
was stopped. download_curl.py fetches identical pinned files via IPv4,
checks expected sizes and available official LFS SHA-256 digests, and
records SHA-256 for every file. Original scripts and logs are retained.

Authoritative success markers: DOWNLOAD_COMPLETED.json for checkpoint
verification, then SMOKE_PASSED.json for all 32 valid returned judgments.
STARTED.json or a live process alone does not imply successful inference.

COMPLETED RESULT

Successful attempt: attempt2_compatible_transformers/. All 32 outputs are
valid and raw-output checksums and paired case identities were verified.
Model memory: 65.28 GiB. Peak total GPU allocation observed by nvidia-smi:
90024 MiB (87.91 GiB), including cache and existing keepalive. GPU returned
to its pre-test 1687 MiB allocation when the inference process exited.
Startup/loading/warm-up: 80.54 seconds.
Hard-label batch: 16 calls, 12553 input tokens, 17.316 seconds.
Probability batch: 16 calls, 13568 input tokens, 1.139 seconds.
Maximum input length: 2219 tokens. The first batch has cold-start overhead;
these two timings cannot establish that probability output is faster, nor
do they establish a reliable full-dataset ETA.

Diagnostic accuracy on the same 16 selected examples:
                   Hard labels     Probability >= 50
Mistral Small 4        8/16               7/16
Qwen 27B             13/16              12/16
Gemma 31B            12/16              13/16
Qwen/Gemma values reuse existing non-reasoning outputs, with zero new calls.
This tiny, balanced mixed-condition sample is NOT the weighted full-test
evaluation. Probability cutoffs were not tuned. The result establishes
technical feasibility but is not sufficient to rank models conclusively.

Dependency repair: vLLM 0.30.0 could not import its Pixtral architecture
against shared Transformers 5.18.0. Its expected PixtralRotaryEmbedding
symbol had been removed. An isolated, checksum-verified overlay of official
Transformers 5.4.0 plus tokenizers 0.22.2 resolves the import. Shared packages
were not modified. Original attempt and logs remain intact. No judge calls
occurred in the failed startup. The repaired attempt uses exactly the same
frozen cases and task messages. Prompt audit verified reasoning_effort=none
in the official Mistral template and no reasoning delimiters in the rendered
assistant prefix. Raw outputs and the full runtime configuration are saved.
