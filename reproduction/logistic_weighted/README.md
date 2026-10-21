# Class-weighted logistic regression

Remote run: `/home/ubuntu/experiments/logistic-weighted-20261001`.

This is a separate ablation of the unweighted run. It uses the exact saved mean-pooled content features and frozen train/validation/test assignments from `/home/ubuntu/experiments/logistic-probes-20261001`. No activation extraction, feature pooling, split changes, or oversampling occurs. Baseline artifacts remain unchanged.

Training uses `class_weight='balanced'`: each example in class c receives weight n_train / (2 × n_train,c). Weights use training labels only and give the two classes equal total training-loss weight. This makes positive examples about 3.53 times as influential as negatives in WildGuard and 12.89 times in ToxicChat. High-Stakes was already approximately balanced.

All other settings match the baseline: training-only StandardScaler, learned intercept, L2 logistic loss, LBFGS, tolerance 1e-6, maximum 2,000 iterations, and C in {0.001, 0.01, 0.1, 1, 10}. Nonconvergence raises an error. For each C, choose the validation-F1-maximizing threshold; select the C with best validation F1. Ties prefer higher thresholds and smaller C. All model selections are frozen before evaluating the test data. Evaluation metrics themselves are unweighted and reflect the original retained class proportions.

Per-dataset split audits contain SHA-256 hashes of the frozen baseline split files and the exact class weights. Saved artifacts include fitted probes/scalers/thresholds, every validation trial, test probabilities and predictions, train/validation/test metrics, and High-Stakes domain breakdowns. `COMPLETED` is written only after all three evaluations finish.

`compare.py BASELINE_DIRECTORY` verifies identical test indices and labels, generates the side-by-side report, and overlays baseline and weighted ROC curves. Both methods' operating points use their own validation-selected thresholds. Sigmoid scores under class weighting are not automatically calibrated probabilities for the original class prevalence.

Focal loss is not used in this run. Its focusing parameter would require a separately specified validation comparison.
