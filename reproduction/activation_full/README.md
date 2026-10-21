# Full activation extraction

Remote directory: `/home/ubuntu/experiments/activation-full-20261001`.

Uses the validated pilot's pinned Llama-3.2-1B-Instruct and dataset revisions, chat formatting, eligible examples, and post-block residual activations at block index 10 (11th block). All non-padding tokens are saved in BF16 without truncation. The full backbone runs without generation, vocabulary logits, or KV cache.

Expected inventory:

| Dataset | Examples | Tokens |
|---|---:|---:|
| WildGuard response harmfulness | 39,643 | 18,310,532 |
| High-stakes training, synthetic held-out, balanced external evaluations | 18,098 | 4,025,903 |
| ToxicChat January 2024 toxicity | 10,165 | 707,627 |

Inputs and official source splits remain as in the pilot. No cross-split deduplication or probe training occurs. Source references and messages permit later grouping, split audits, and identification of relevant token spans. Labels and annotations are not model input.

Examples are sorted by length within each dataset, processed eight at a time, and saved in shards of up to 64 examples. Each safetensors file contains unpadded `activations`, `token_ids`, and `offsets`; its companion JSON records labels, source IDs, original dataset indices, messages, and a checksum. `manifest.json` lists all shards after completion.

Each new shard is checked for shape, BF16 dtype, finite values, and exact equality after disk read-back. Files are flushed and committed atomically. Status is updated after every committed shard. Existing complete shards can be resumed after checksum, configuration, and sample-identity verification. An unresolved partial or unpaired file stops the run for inspection; there are no automatic retries. A process lock prevents concurrent writers.

To inspect later, read `status.json`, `journal.jsonl`, and `run.log`, then check the PID in `pid.txt` and GPU processes. `COMPLETED` is written only after all expected examples/tokens and shard totals reconcile. Exceptions write `FAILED.json`. Preserve all files and the unrelated keepalive workloads.

The process is launched detached with `nohup`; closing this task does not stop it. It does not restart itself after an instance stop. No scheduled monitor was created.
