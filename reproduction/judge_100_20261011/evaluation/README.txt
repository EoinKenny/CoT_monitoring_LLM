Seven-judge evaluation adapter
=============================

Run only after collection has completed all cells:

  /home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python \
    /home/ubuntu/experiments/judge-expansion-20261011/evaluation/evaluate_expansion.py \
    --hard-root /home/ubuntu/experiments/judge-expansion-20261011/hard \
    --probability-root /home/ubuntu/experiments/judge-expansion-20261011/probability \
    --output /home/ubuntu/experiments/judge-expansion-20261011/analysis

Add --verify-only to check collection coverage without running estimation.
No inference, retraining, or sampling occurs in this adapter.

Required directory structure is exactly the historical full-threshold and
confidence-threshold sweeps. Existing qwen/gemma judge cells may be symlinked
to authoritative completed cells; this script only reads those cells.
Do not symlink the hard/probability root-level RESULTS.json, COMPLETED.json,
EVALUATION_COMPLETED.json, or per-backbone evaluation directories to historical
locations because this script writes new expanded evaluation artifacts there.
All five backbone directories require selected_flags.json, SAMPLE_FROZEN.json,
THRESHOLDS.json, the ten *_predictions.npz files, and hard/{b}/judge_inputs/
explanations.jsonl. Probability and hard frozen metadata must match bytewise.

Each cell b/judges/{judge}_{condition} has prompts.jsonl plus run/config.json,
run/judgments.jsonl, run/COMPLETED, and run/batches/*.json with *.raw.json.
The ordered 10,000 sample IDs per cell must exactly equal selected_flags.json.
Fields consumed from judgment rows are sample_id, prediction (hard only),
positive_probability (probability only, integer0..100), input_tokens,
batch_number, batch_seconds. Preserve additional provenance and raw outputs.

The original hard-label evaluator, probability evaluator, common module,
comprehensive analyzer and summarizer are copied unchanged into upstream/.
UPSTREAM_SHA256.json pins those bytes. Runtime modifications are restricted to
paths, the seven judges and expected completion counts. Mathematical functions,
threshold grids, tie rules, random seeds, design weights, bootstrap protocols,
and validation/test membership remain unchanged. Input cases are not pooled
across backbones and are paired across judges/conditions within a backbone.

Outputs:
- hard/RESULTS.json:350 pairs,3 conditions with validation-selected probe gate.
- probability/RESULTS.json:350 pairs,9 hard/fixed50/joint variants.
- analysis/:full historical comprehensive analyses across350 pairs.
- analysis/summary.json and judge_summary.csv:overall/per-judge macro metrics,
  heterogeneity and original paired dataset-block bootstrap contrasts.
- analysis/input_token_costs.csv:design-weighted input-token estimates per
  population example, per1,000 examples and per referred case at each policy's
  validation-selected gate. The token metric is descriptive and is not used
  to change the original policy selection or threshold tie-breaking.
- analysis/report.html:neutral tabular report, without assumed NN superiority.

Input-token cost does not use the judge output probability cutoff: a routed
case costs input tokens even if the judge ultimately rejects it. Counts are
uncached tokenizer-specific input lengths, not a latency or monetary model.
All raw calls are counted separately in the existing runtime.json.

Run tests with any Python environment containing NumPy:
  python -m unittest discover -s evaluation -p test_adapter.py -v

Completion checks require all2,100,000 outputs, exact paired IDs and metadata,
raw/commit/config/prompt hashes, finite valid bounded output values, sampled
within-dataset split-group separation and the20GiB free-space reserve. Numeric
activation replay agreement gates are neither restored nor introduced.
Existing failed or ambiguous attempts are retained by the collection runner.

Scientific interpretation is unchanged: pipeline metrics estimate complete-test
populations from the stratified samples. Probe-only metrics are exact. Thresholds
are tuned on validation only. The study already has test inspection and remains
exploratory; no guarantee of general safety gains or NN superiority follows.
