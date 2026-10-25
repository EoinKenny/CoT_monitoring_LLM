# Fresh model collection

The offline plotting command never queries a model. `--query-models CONFIG`
explicitly collects fresh judgments, using the frozen prompts and case/split
membership. It creates a separate SQLite request journal and raw-response store.
It does not promise identical new outputs from changed provider versions.

1. Copy `configs/collection.example.json` to `configs/my_models.json` (gitignored).
2. Set an exact backend model ID, endpoint URL and native-thinking settings for
   each judge. Local servers can use the OpenAI-compatible Chat Completions API.
   A hosted example is in `configs/hosted.example.json`.
   Hosted APIs use `provider: "openai-compatible"` or `"anthropic"` and
   `api_key_env` naming an environment variable; never put a key in the file.
3. Specify `trace_labeler` as a configured alias (Qwen122 in the paper). All
   configured models annotate the original target/NN pair; the named trace labeler
   separately annotates the written explanations.

```bash
python reproduce_results.py --query-models configs/my_models.json --dry-run
python reproduce_results.py --query-models configs/my_models.json --studies main --limit 8 --out outputs/smoke
python reproduce_results.py --query-models configs/my_models.json --out outputs/new_collection
```

Default studies: main, 2-NN, ablation, pair mechanism labels, probability-first
explanations, trace labels. Add `reason_first` to `--studies` for the archived
512-token explanation-first condition. Without `--limit`, reasoning follows each
model's **own** valid control-versus-1NN test flips at the common cutoff 50; it does
not borrow another model's selections. `--limit` applies per model/study and is
for smoke tests, not full evaluation.

Static per-model calls: 150,000 main; 50,000 2-NN; 350,000 ablations (validation
included in the frozen catalogue, though the ablation plot uses test only);
25,000 pair annotations. Reasoning and trace-label counts depend on valid flips.
The compact 42 MiB prompt catalogue restores to about 614 MiB on first collection.
The runner preserves at least 20 GiB free disk.

Temperature is 0. Output caps: classification 8, pair/trace labels 128,
probability-first 256, explanation-first 512. Provider parameter support varies:
set `token_parameter` to `max_tokens` for servers that use it, or
`max_completion_tokens` where required. Use `extra_body` for the provider's
explicit native-thinking disable setting; protocol fields cannot be overridden.
Errors are retained rather than changing the settings automatically.
See [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [Claude stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons).

## Outputs and resume

`responses.sqlite` contains exact raw response JSON/text, parsed values, status,
finish reason, returned model ID, input/output tokens, timing, message digest, and
case mapping. `data/paper/analysis/case_identities.json` maps case keys to
`[backbone, sample_id, dataset]`, except for mechanism prompts, where the middle
value is a `unit_id` (also the prompt ID). Resolve that ID through
`pair_mapping.jsonl.gz` to obtain its original sample ID. All 625,000 prompt
mappings were checked against the frozen design. The configuration record
contains only environment-variable names,
not API keys. A dispatch transaction is committed before every network request.
A crash or timeout may leave a dispatch ambiguous; resuming skips **all attempted
requests**, including invalid/error/ambiguous ones. No retries or output replacement.
Only one controller can hold the output directory lock. Resume with the identical
configuration and output path. Use a new directory for a changed configuration.

`main_scores.npz`, `ablation_scores.npz`, and `COVERAGE.json` are exported after
collection. Invalid/missing scores remain -1. Pair/trace JSON labels and trace
source mappings remain in the same database. New collections are separate from
published plotting inputs: they are new experiments, not a patch to frozen data.

The frozen analysis accepts the user's final-integer parsing amendment for
classification: use a standalone final integer 0–100 in a normally completed
response, while retaining all preceding text. Probability-first explanations use
the first standalone probability, even when the explanation is truncated or
contradictory. No internal reasoning is inferred from written explanations.
