# Full released-paper baseline

This setup keeps four stages separate:

1. Download and pin the public Hugging Face rollout snapshot.
2. Inventory every released rollout directory and detector, with hashes.
3. Run model-compatible probes over each dataset with an 8,192-token safety ceiling; record every skipped dialogue.
4. Reconstruct the ten paper threshold operating points using only the retained development split, then freeze the resulting manifest for future tests.

The paper's primary GLM family contains 18 indicators at layers 27 and 29. The separately released Qwen cross-model family is inventoried and evaluated only against compatible Qwen rollouts. Test and OOD labels are never used to tune thresholds.

The permanent threshold record must include the source repository commit, data snapshot revision, probe hashes, token ceiling, per-dataset retained/skipped counts, development split, score aggregation, and thresholds for every layer and FPR target.

For the probe-to-LLM cascade, `run_mixed_qwen_binary_vllm.py` uses the
released `probe_llm_cascade.py` binary prompt and parses its
`has_misalignment` verdict. The earlier 1--10 score experiment is retained as
a separate sensitivity analysis; it is not the paper-faithful cascade.
