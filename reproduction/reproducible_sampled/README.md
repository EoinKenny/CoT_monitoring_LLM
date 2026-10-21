# Reproducible sampled evaluation — 2026-10-04

Three fixed activation backbones, ten binary tasks, five existing judges and three conditions (baseline, highest-mean-logit sentence, nearest training neighbour). Prior experiments remain unchanged. The new remote root is `/home/ubuntu/experiments/reproducible-sampled-20261004`.

## Fixed protocol

All eligible training examples; balanced logistic-regression class weights; no undersampling/PCA. StandardScaler fits train only. L2 C=.01 (HighStakes .001), lbfgs, max_iter=2000, tol=1e-6. Validation threshold maximizes positive F1; thresholds within 1e-12 F1 tie use the highest threshold, matching previous implementation. Existing source-group splits, exact-input filters and Llama 4096-token eligibility remain fixed. New backbones never truncate these examples.

Full-training method selection on FAVA used test F1 and is exploratory; this does not establish universal superiority. Earlier inspection of these test sets also limits confirmatory claims.

Each backbone/dataset selects up to 500 probe-positive test cases, targeting TP=250 and FP=250, filling a scarce stratum's unused quota from the other. Sample all flags if fewer than 500. Sort SHA256 UTF-8 NUL-joined seed (20261004), backbone short ID (llama/qwen/gemma), dataset ID, stratum (TP/FP), immutable ID; ties use ID. Immutable ID is dataset plus SHA256 of compact JSON `[string source ID, input_hash]`. Gold is used only for sampling/evaluation, never as a target label in judge prompts. Every judge/condition sees the same ordered selection within each backbone.

Pinned weights, historical content masks and layer outputs are retained. Gemma keeps its documented tool/channel-content rendering adaptation and exact 41-layer causal prefix with CPU embeddings. Missing training vectors use BF16, fixed batches sorted by length/ID, fixed seed, deterministic PyTorch operations and CUBLAS_WORKSPACE_CONFIG. Existing activation artifacts remain authoritative; numerical agreement across software/hardware or extraction batching is not claimed bitwise. Each stage records its environment. Newly extracted vectors are committed in checksum-verified shards and merged with unchanged cached vectors.

NN retrieval uses cosine similarity of training-standardized pooled vectors across the entire training pool, with exact chunked search and ties by immutable ID. Saliency uses PySBD English sentence spans and the new probe's token logits; pooled logits must match sequence logits within 1e-4. Each target keeps its full conversation. All prompts and configurations are saved and hashed.

## Judgment and recovery

Existing five models, quantization modes and prompt wording are retained. vLLM greedy temperature=0, one restricted 0/1 token, seed=20261001, eager, no prefix caching. New judgments are generated afresh (no historical reuse assumed). This conservative policy avoids mixing changed probes/explanations or model/runtime configurations. vLLM is not claimed bitwise reproducible.

One GPU process at a time, alongside existing keepalives. Before each batch, write an attempt record. Preserve returned raw generation, then atomically commit validated judgments with checksum. Restart skips verified committed batches; an attempt without a valid committed result blocks automatic inference retry. Investigate and document ambiguous attempts instead of silently regenerating. Activation extraction is deterministic and can replay uncommitted work. Other partial stages stop for inspection. Check the 20 GiB reserve at stage and batch boundaries.

`supervise.py` is detached and holds a filesystem lock. It freezes code checksums, trains/selects all backbones, builds all explanations/prompts, records workload/ETA, then runs all judge conditions and reports. Logs and status identify the active stage. Resume by running the same supervisor; no edits to frozen code without an explicit documented revision. `COMPLETED` is only written after all 450 evaluation cells pass coverage checks.

## Evaluation

Full-test probe metrics are exact. Selected-flag metrics describe the stratified sample. Full-test cascade confusion counts estimate retained TP/FP with weights N_stratum/n_stratum, adding exact unflagged FN/TN; calculate F1 from these counts. Label as estimated unless all flags were selected. Do not average stratum F1. Report counts, per-dataset results, macro means and paired saliency/NN versus baseline changes. Paired within-stratum bootstrap uses 2,000 draws, keeps census strata fixed, and uses identical resampled cases for both conditions. Intervals describe sampling uncertainty conditional on fitted probes and the fixed test population; ordinary bootstrap ignores finite-population correction and is approximate/conservative at large sampling fractions. They do not measure population generalization.

## Artifacts and commands

`code/test_protocol.py` checks reordering invariance, balanced/imbalanced/empty/census cases, context-dependent sampling and deployment weighting. `SAMPLING_BALANCE.json` reports every shortfall. `ETA.json` uses historic measured generation throughput and newly measured input-token totals. `RESULTS.json`, `judge_results.csv`, `probe_results.csv`, `RESULTS.md`, and three 10-panel ROC figures summarize completed runs. Failure records and raw outputs are never discarded automatically.

Run the supervisor with the existing CPU Python after preparation is complete; it selects the original CPU/Transformers environment for Llama/Qwen and the existing vLLM environment for Gemma/judges. Source adapters are derived with explicit replacements from historical scripts, with original/adapted checksums in `ADAPTER_SOURCE_MANIFEST.json`. Cleanup manifest names every approved redundant weight object and pip-cache file removed; no other experiments or active weights are removed.
