# Seven additional datasets: balanced backbone feasibility pilot

Remote root: /home/ubuntu/experiments/seven-dataset-pilot-20261002
Purpose: test the existing activation backbone and binary probe recipe on seven new dataset families before scaling the full judge/explanation cascade. This pilot does not invoke a judge or measure deployment prevalence.

Frozen Llama-3.2-1B-Instruct revision 9213176726f574b556790deb65791e0c5aa438b6, block index 10 (11th block), BF16. Mean content activations; padding and template/special tokens excluded. Input tasks pool user content. Response tasks pool assistant content conditioned on the request/evidence. No PCA. StandardScaler fitted only on training data, L2 logistic regression C=0.01 and class_weight=balanced; validation F1 threshold, highest threshold breaks ties. No hyperparameter sweep or test tuning.

400 train / 100 validation / 200 test per dataset, exactly 50/50 classes in every split. Fixed seed 20261002. Preserve held-out official test sources where present, sample test subsets; use group-hash partitions for datasets without official splits. Related source groups and exact inputs cannot cross splits. Conflicting identical inputs removed. BeaverTails prompt groups shared with official tests are excluded from training. FAVA groups union rows sharing identical reference passages. HaluEval groups by supplied evidence, RAGTruth by source_id, WildJailbreak by vanilla request, ToxiGen by normalized text. Duplicate checks are exact/whitespace-normalized, not a semantic near-duplicate guarantee.

Binary positive definitions:
- BeaverTails: not is_safe; response safety given prompt. Original 30K split only.
- AEGIS 2.0: prompt_label == unsafe; exclude REDACTED/invalid/empty prompts. Main split files only, not auxiliary refusal files. Official validation plus held-out groups from training form the validation candidate pool.
- ToxiGen: mean human toxicity rating >3 on 1–5 scale; <=3 negative. Annotated subset only. This is a declared binary adaptation, not the generation label.
- HaluEval-QA: provided hallucinated_answer positive; right_answer negative. The answer's label is never put in its input. Original questions/evidence govern grouping before selection.
- RAGTruth: at least one annotated span not marked implicit_true; good-quality responses only; full source prompt included.
- FAVA: original passage has a recognized error-category annotation. Reconstruct original text by removing mark-tag contents (proposed replacements), keeping delete-tag contents (original errors), then stripping annotation tags. Require exact occurrence in published input prompt. Evidence separated before the error-identification instruction. No-error passages are existing unannotated completions, not newly edited counterfactuals. Exclude unknown tags and reconstruction mismatches. Human benchmark is not used here; this is synthetic train/val/test feasibility.
- WildJailbreak: data_type harmful vs benign, including vanilla/adversarial; do not use response, tactics, or label in model input. Official test is adversarial.

Inputs longer than 4096 tokens are excluded, never truncated; none of the candidates scanned for the selected pilot hit this limit. Final maximum lengths are recorded per dataset. Activations and pooled vectors, exact selected rows, source revisions, raw failures, and predictions stay on the instance. Only aggregate results should be retrieved to local.

A length-only classifier using log content-token count and log total-token count is fitted on the same training data and thresholded on validation as a dataset-shortcut diagnostic. Evaluation uses the same fixed test subset; it is not a replacement model selected using test scores.

WildJailbreak's first download failed because account access was gated; that log is preserved. User accepted access and subsequent authorized download succeeded. No data-access gate was bypassed.
