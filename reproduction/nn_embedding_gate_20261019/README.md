# Embedding-based nearest-neighbour explanation selectors

This study reuses the frozen original target and training-neighbour activations, cases, judge answers, splits and sampling weights. It makes no new judge calls.

Primary analysis: Aegis, BeaverTails, ToxicChat and ToxiGen; five backbones and ten judges; probe/judge thresholds fixed at 0.5/50. A selector is fitted for each backbone–judge pair, pooling the four datasets. It chooses whether to use the existing NN answer or control answer. Training labels are available only on validation disagreements, where exactly one answer is correct. Inclusion cutoffs are selected separately per dataset from out-of-fold validation predictions.

The numerical-only matched baseline has the same grouping and validation procedure. Historical pooled numerical and cosine selectors are replayed from existing frozen predictions. This distinguishes the effect of features from the effect of training-pool size.

## Files and execution order

1. `export_embeddings.py`: match immutable original target/neighbour identities and export cached activations.
2. `prepare.py`: fold-fitted and full-validation PCA and feature-only transformations.
3. `train.py`: grouped validation fitting, candidate selection and full-validation refitting.
4. `controller.py`: CPU orchestration with process lock and integrity receipts.
5. `evaluate.py`: require all training receipts, freeze selections, evaluate paired held-out cases and replay historical metrics.
6. `analyze.py`: paired uncertainty, macro/stratified results and report.

`evaluation_controller.py` waits for training to complete and invokes evaluation once under a process lock. It does not retry training or inference.

Remote experiment: `/home/ubuntu/experiments/nn-embedding-gate-20261019`.

Full candidate specification is in `PROTOCOL.json`. All original inputs have SHA256 hashes in `INPUTS.json`. Export, representation, training and evaluation stages retain separate completion receipts. Exact fitted model binaries and embedding arrays remain on the experiment server; compact results are mirrored locally.

## Interpretation limits

PCA and model preprocessing are fitted within validation training folds; no test outcomes select the representation, classifier or cutoff. However, this project's test set has already been examined in earlier research, so this is exploratory evidence, not a fresh confirmation set.

The secondary three-anchor comparison reuses architectures selected on fixed-50 validation, then reselects C/k and inclusion cutoffs for each anchor. It does not repeat the whole architecture search independently under each anchor.

Conditional paired intervals use 2,000 target-cluster exponential multiplier replicates shared across judges/backbones. Weights are calibrated within each original sampling stratum; census strata remain fixed. This retains the original population denominators for local judges and recomputes retained weighted support for API judges. Intervals do not include training/model-selection uncertainty or multiple-comparison correction. Candidate families are all reported, including negative results.
