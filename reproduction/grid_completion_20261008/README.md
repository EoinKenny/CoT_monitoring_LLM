# Complete the five-backbone, six-judge local grid

Authorized 8 October 2026. Remote root:
`/home/ubuntu/experiments/grid-completion-20261008` on the G7e.

## Scope

The existing Llama-3.2-1B, Qwen3-4B, Gemma31B and Llama70B backbone
experiments contain 60 completed judge runs (600 dataset/condition cells).
`prepare_grid.py` checks their output checksums, coverage and selected IDs,
then links their authoritative artifacts without rewriting their results.

Qwen3.5-122B-A10B-FP8 has 456,008 completed pooled activation vectors and no
downstream results at launch. Train its ten probes from scratch, select flags,
derive explanations, and run the five established judges. Add official
`Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` as a sixth judge across all five backbones.
The target is 900 local judge/dataset/condition cells plus 50 probe summaries
and their train/validation/test metrics. API results remain separate: this
controller does not call external APIs or fabricate missing API responses.

## Protocol retained

- All eligible training examples, `class_weight='balanced'`, no undersampling
  or PCA. Train-only StandardScaler; logistic regression C=0.01, except
  high_stakes C=0.001, lbfgs, max_iter=2000, tol=1e-6, seed 20261004.
- Preserve frozen train/validation/test memberships and source groups. Choose
  maximum validation F1, highest threshold on ties using the established
  1e-12 numerical tie tolerance. Test metrics do not choose new hyperparameters.
- Up to 500 flagged cases per backbone/dataset, targeting 250 TP and 250 FP,
  filling shortages from the other stratum. Use the established SHA-256 rank
  with seed 20261004 and immutable IDs. Save population counts and weights.
- New backbone sampling key: `qwen122`. All six judges use exactly its same
  frozen selected cases. The new judge uses each older backbone's already
  frozen cases; it does not resample them.
- Saliency: replay the original extraction batches with the original FP8
  prefix runtime, requiring bitwise equality to the pooled vectors. Use the
  highest mean probe-logit sentence over content tokens, PySBD English spans,
  the saved literal content spans and the documented empty-user template adapter.
  Fail on replay differences, rather than silently changing explanations.
- Neighbours: exact cosine similarity of standardized activations over the
  complete training pool; immutable training ID breaks ties. Show the neighbour's
  gold label and probe prediction, never the target example's gold label.
- Preserve the established rubrics and condition wording. Native judge chat
  templates, thinking disabled, greedy one-token classification restricted to
  0/1, seed 20261001. No input truncation. Freeze prompt IDs, text/token hashes,
  model revisions, package versions, hardware and engine settings.
- Report exact full-test probe metrics, sampled-flag metrics, weighted full-test
  pipeline confusion counts/F1, and paired stratified bootstrap comparisons.
  Preserve the historical caveat that prior pilot/test inspection, including
  FAVA training-policy selection, limits confirmatory interpretation.

## Quantization and resource distinction

The new backbone features remain the official FP8 activations after block 31.
The new judge uses all 48 blocks of the official GPTQ 4-bit checkpoint; unlike
prefix extraction, full FP8 weights exceed this GPU's memory. It is still a
122B-total / approximately 10B-active MoE, not a dense 122B model. Record judge
precision explicitly in comparisons. Its pinned revision and per-file checksums
are under `qwen122_judge_download`. The weight cache is on the NVMe model disk.

The full judge must pass a separate engine pilot spanning shortest and longest
prompts across all three conditions. Pilot outputs are retained separately and
are not inserted into the canonical evaluation. The five established judges
run first, so their missing-backbone results can finish independently of the
new large judge's deployment. GPU jobs are serial within the controller.

## Recovery and artifacts

`controller.py` runs detached. `STATUS.json`, `CHILD.json`, per-stage logs and
completion markers identify real progress. Frozen source checksums are in
`CODE_FROZEN.json`; imported result proofs are in `IMPORT_AUDIT.json`.
Keep 20 GiB free on the main disk. Stop on a failed stage or ambiguous inference
attempt; preserve diagnostics and raw generations. Verified committed batches
can resume without regeneration. Fresh vLLM inference is not promised bitwise
identical across hardware, even with the same seeds and settings.

Canonical output: `RESULTS.json`, `judge_results.csv`, `probe_results.csv`,
`RESULTS.md`, five backbone ROC figures, and per-backbone sampling/evaluation
manifests. `COMPLETED` plus completed status confirms the entire 900-cell grid;
running processes or partial files alone do not.
