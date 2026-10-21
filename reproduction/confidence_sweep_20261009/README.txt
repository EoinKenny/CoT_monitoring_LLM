Probability-only threshold study

Parent: /home/ubuntu/experiments/full-threshold-sweep-20261009
Run: /home/ubuntu/experiments/confidence-threshold-sweep-20261009

Five backbones, ten datasets, two judges, three explanation conditions.
Each judge receives the exact parent cases in the same order, with the same
conversation and explanation bytes. Only the response contract and final
response cue change. Output is one integer 0..100 for positive-class probability.

The isolated smoke runs cover 32 length-quantile examples per explanation
condition per judge, including shortest and longest prompts across backbones.
Their predictions are not reused by the main experiment. All attempts and raw
generations are retained. Incomplete or malformed attempts block resume.

Inference: greedy, seed 20261001, batch 16, max_num_seqs 16,
max_num_batched_tokens 4096, max_tokens 8, xgrammar structured choices
containing the complete strings 0 through 100. This is not an allowed-token-ID
filter. Native parent chat templates and frozen local tokenizer/model revisions
are retained. Model context length is rounded up to fit new prompts plus output.
Fresh inference can vary despite matching settings; replay saved artifacts for
exact output reproduction. Model probabilities are verbalized, not calibrated.

Evaluation:
- Parent hard-label pipeline at its validation-selected probe threshold.
- Probability-only pipeline with cutoff 50, validation-tuned probe threshold.
- Probability-only pipeline with jointly validation-tuned probe/score cutoffs.
- Parent probe-only baseline selected on complete validation and evaluated on
  complete test data.
The label is derived downstream; there is no separate redundant label output.

Joint cutoff grid: parent validation-derived probe thresholds x integers 0..101.
101 rejects every probability output. Predict positive iff BOTH scores meet
thresholds. Maximize weighted validation F1; ties within 1e-12 choose highest
probe threshold, then highest probability threshold. Freeze before test scoring.
Test curves are exploratory; no test-optimal threshold is reported as held-out.

Pipeline confusion counts use inverse inclusion weights and exact total class
populations. Probe-only and referral rates use all eligible test predictions.
Pipeline estimates are labelled as such whenever sampling was used. Uncertainty:
2000 matched, stratified test bootstraps, census strata fixed; thresholds are
fixed at their validation choices. Intervals are conditional and pointwise, not
simultaneous confidence bands or estimates of validation-selection uncertainty.

Artifacts: PROTOCOL.json, PARENT_VERIFIED.json, PREPARED.json, CODE_FROZEN.json,
ENVIRONMENT.json, smoke outputs, exact prompts, run/config.json, durable raw and
committed batches, judgments.jsonl, RESULTS.json, per-pair validation choices
and threshold curves. Parent artifacts are never overwritten. The controller
stops below 20 GiB free disk and does not silently retry ambiguous inference.
