# Mean-pooled logistic probes

Remote run: `/home/ubuntu/experiments/logistic-probes-20261001`.

One binary probe is fitted for each dataset using the frozen, pinned Llama-3.2-1B-Instruct activation cache from `/home/ubuntu/experiments/activation-full-20261001`. Activations are the residual output of block index 10 (11th block).

## Pooling

Each saved shard's SHA-256 is checked. The pinned tokenizer reconstructs the full chat sequence and must reproduce every cached token ID exactly. Character offsets identify supplied message content. All special token IDs and template-only positions are excluded, including automatically inserted system boilerplate. Tokens overlapping the supplied content span are retained; boundary whitespace can share a token with content. Padding was already removed during extraction.

- WildGuard: assistant response content only; those activations retain preceding prompt context.
- ToxicChat: user content only.
- High-stakes: all supplied message content, including supplied system context.

Selected BF16 activations are converted to FP32 and averaged once per example. Empty-content examples are excluded explicitly. The example label supervises one prediction; tokens are not independently assigned that label.

The template date is pinned to the extraction date, `01 Oct 2026`, rather than the current date. Tool-output strings are JSON-escaped by the model's chat template: their rendered body is round-trip checked against the original string, and its outer template-generated quotes are excluded. This mapping passed an audit of all 66 cached High-Stakes conversations containing 118 tool messages. The initial simple verbatim-span lookup stopped before any model fitting; that failed attempt is preserved, and the completed WildGuard pooled cache was reused after the correction.

## Splits and model selection

Released test partitions are retained subject to the explicit duplicate/group exclusions below. WildGuard and ToxicChat receive a deterministic approximately 20% validation holdout from their released training partitions using stratified group splitting. High-stakes uses its released synthetic training partition, external validation partitions, and synthetic plus external test partitions. Its synthetic and external test metrics are also reported separately.

Groups connect identical full inputs, WildGuard prompts, ToxicChat conversation IDs/user texts, and high-stakes source IDs, identical user content, and synthetic pair IDs. Conflicting labels for identical full inputs are excluded. Exact duplicates retain one example, preferring test over validation over training. Other groups crossing released partitions retain only their highest-priority partition. Split membership and every exclusion reason are saved. These checks do not claim to detect every semantic near-duplicate.

An additional read-only audit of the released high-stakes synthetic data found no overlap in scenario IDs across train/test (5,695 training IDs, 1,461 test IDs) and no pair-ID overlap.

StandardScaler is fitted on training examples only. Logistic regression uses L2 regularization, a learned intercept, no class weighting, LBFGS, tolerance 1e-6, and a 2,000-iteration limit. Nonconvergence stops the run. Candidate C values are 0.001, 0.01, 0.1, 1, and 10.

For each C, the threshold is selected over all unique validation probabilities to maximize positive-class F1. Equal F1 prefers the higher threshold. The C with best validation F1 is selected; equal F1 prefers smaller C. Models are not refitted using validation labels. All three selections are written to `SELECTIONS_FROZEN.json` before test metrics are calculated.

## Outputs

Per-dataset pooled features, metadata, split assignments, validation trial results, fitted scaler/probe/threshold, and held-out prediction files are retained. `results.json` includes accuracy, precision, recall, F1, AUROC, average precision, confusion counts, positive-call rate, a majority-class accuracy baseline, and a separately reported 0.5-threshold comparison.

`status.json` and `journal.jsonl` track progress. `FAILED.json` records exceptions. `COMPLETED` is written only after all three held-out evaluations finish. This run trains probes only; it does not call an LLM judge.
