# Ten-dataset full balanced-training pipeline

Authorized 2026-10-02: expand training, validation-tune probe thresholds, estimate flagged workload, and run all three judges across ten datasets. This is a new experiment; previous pilot and three-dataset outputs remain intact.

Remote root: /home/ubuntu/experiments/ten-dataset-full-20261002
Probe process at launch: 354238. Downstream supervisor: 354923.

## Scope and labels
The seven new adapters, class definitions, source revisions, group assignments, masking and 4096-token no-truncation eligibility rule are inherited from reproduction/seven_dataset_pilot. The selected BeaverTails corpus is the original human-annotated 30K release, ToxiGen the annotated subset, and FAVA the released synthetic training corpus; full means all eligible examples in those approved source versions, not every alternative release. No FAVA corrected negatives are fabricated.

Training is undersampled to the minority-class count with seed 20261002. Use the full eligible validation/test pools at their available class proportions. These include previously inspected pilot test cases; report this as expanded exploratory evaluation, not wholly fresh confirmatory testing. The original three probes are also retrained on balanced subsets of their frozen original training splits; original validation/test assignments and cached features are retained. Preserve earlier weighted-probe results as a separate comparison.

Backbone: pinned Llama-3.2-1B-Instruct revision 9213176726f574b556790deb65791e0c5aa438b6, block index 10. BF16 activations, FP32 content means, StandardScaler fit on training only, no PCA. Fixed L2 C=0.01 except retained HighStakes C=0.001; class_weight=balanced. Validation F1 selects threshold; highest threshold breaks ties. Test data are not used for tuning. Test flags are precisely TP+FP at that threshold.

Training/validation pooled vectors retained for fitting and exact within-dataset training-only nearest-neighbour retrieval. Full token activations retained for new test cases only, with no padding, for saliency. Original three reuse their existing all-token cache. Group and input-hash intersections are audited across splits. This does not establish semantic near-duplicate removal or cross-dataset independence.

## Judges
Qwen3.6-27B-FP8, Gemma-4-31B-it-qat-w4a16-ct, and the verified AWQ Llama-3.3-70B-Instruct checkpoint use their previously pinned revisions and native chat templates. All three conditions are rerun freshly for all flagged test examples: no explanation, highest-mean-probe-logit sentence, and nearest training example with reference label and probe prediction. No negative probe decisions are changed. Greedy one-token 0/1 classification, seed 20261001, no generated reasoning, vLLM. Standardized pooled-activation cosine selects nearest neighbours; the reference is from the balanced training subset, never validation/test. PySBD defines sentences; saliency only considers eligible target content tokens. Semantic prompt hashes are checked across models. New dataset rubric adaptations are stored in judge_rubrics.json; reference labels are shown only for training neighbours.

CPU/GPU pipeline: wait for probe completion; audit; estimate runtime from actual baseline token counts and measured previous condition runtimes; extract explanations; prepare exact model-specific prompt token IDs; verify semantics; run nine model/condition jobs sequentially; evaluate each completed condition. No automatic retry after errors and no overwriting prior judgments. Logs are durable and failures visible through PIPELINE_STATUS.json. Workload estimates use a broad range because longer reference contexts can change throughput.

## Status and artifacts
Probe results: results.json, COMPLETED, VERIFIED.json. Workload: workload_estimate.json. Pipeline: PIPELINE_STATUS.json, pipeline.log, PIPELINE_COMPLETED. Judge outputs: judges/<model>_<condition>/run/{status.json,judgments.jsonl,results.json,COMPLETED,FAILED.json}. Scripts and aggregate reports may be copied locally. Raw selected texts, references, prompts and judgments remain on the GPU instance.

Cascade reports include balanced accuracy and macro-F1 alongside positive-class F1 to expose majority-class effects, particularly on FAVA and WildJailbreak.
