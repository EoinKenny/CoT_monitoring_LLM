# All-validation three-class NN selectors

Train on every eligible validation row, with helpful (+1), harmful (-1), and unchanged (0) outcomes defined from the existing paired pipeline decisions and validation gold labels. Unchanged includes both-correct, both-wrong and nonreferred cases. The target label is never an input feature.

Scope: Aegis, BeaverTails, ToxicChat and ToxiGen; five backbones; ten judges. One selector per backbone–judge pair pools the four datasets. Compare fixed probe/judge 50/50 and the three previously frozen shared threshold anchors. Preserve design weights, paired-valid masks, target-grouped folds and population denominators.

Keep the previous validation-selected representation and learner for each of five families fixed; reselect C or k and inclusion thresholds on validation. The previous disagreement-only winners remain within these architecture sets. This isolates the training-label treatment without a new architecture search. Three-class logistic uses normalized one-vs-rest probabilities; k-NN uses weighted class proportions. Inclusion score is P(helpful) minus P(harmful), rescaled monotonically to [0,1] for the existing cutoff routine. No class rebalance is applied.

The parent cached features, PCA transformations, original inputs and benchmark predictions are read from `/home/ubuntu/experiments/nn-embedding-gate-20261019`. They are not modified. Models and new results live in `/home/ubuntu/experiments/nn-three-class-gate-20261019`.

- `common.py`: frozen-data readers, feature representations, three-class learners and probability-to-utility scoring.
- `train.py`: all-validation grouped CV, selection and full-validation refits, class labels/counts, checksummed checkpoints.
- `controller.py`: three CPU workers; freeze and evaluate only after every group finishes.
- `evaluate.py`: checksum verification, frozen choices, original benchmark replay, paired test evaluation.
- `analyze.py`: compare immutable disagreement-only outputs, paired target-cluster uncertainty, judge/dataset/backbone tables and report.

Synthetic smoke checks cover class-probability normalization, signed utility, both k-NN metrics and constant-class cases. No new LLM calls are made. Prior outcomes and failed/partial training records are preserved.

Interpretation: the test set has been previously explored. The new three-class models do not receive a fresh full representation search; the representations were selected earlier for disagreement-only training. Paired intervals are conditional on fitted models and exclude selection uncertainty. Large unchanged-class counts must not be interpreted as equally many new examples of where NN helps or hurts.

## Numerical implementation amendment

Remaining high-dimensional logistic fits use double-precision row-space coordinates for the same L2-regularized objective, with coefficients mapped back to original feature coordinates before storage. Only numerical-null directions are removed at a relative eigenvalue tolerance of 1e-12; no low-rank representation choice is introduced. Original completed fits remain untouched. The remote IMPLEMENTATION_AMENDMENT.json records the preserved interrupted attempts, solver benchmarks, and resumption provenance. Model probability and serialization smoke checks passed.

Subsequent implementation amendments use the equivalent primal solver in row-space coordinates, cache these coordinates by the complete standardized training matrix SHA256, and avoid materializing a large rotation matrix during coefficient backprojection. Cache entries depend only on validation training-fold features, never labels or held-out data. Parent architecture files are independently verified in PARENT_PROVENANCE.json.
