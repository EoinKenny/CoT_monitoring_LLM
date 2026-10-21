# Fresh-trained deception probe baseline

This pipeline resumes/trains eight deception-focused layer-27 linear probes from
the paper's public indicator training data, validates their held-out thresholds,
scores the same mixed DeceptionBench/hard-negative test, and escalates every
probe-positive turn to a local Qwen hard-classification judge through vLLM.

The two already completed fresh checkpoints are reused. Thresholds are each
probe's held-out turn-level 1%-FPR operating point; test labels never tune them.
GLM sequences beyond 8,192 tokens and Qwen prompts beyond 32,768 tokens are
skipped and counted in the output artifacts.

`run_f1_tuned_cascade.sh` additionally selects one shared multiplier on the
eight fixed validation thresholds to maximize ensemble F1 on the labeled test
set, then reruns the binary Qwen cascade. This is intentionally an in-sample,
reproducible operating point and is labelled as a resubstitution result rather
than a generalization estimate.
