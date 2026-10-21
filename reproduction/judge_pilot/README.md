# Qwen hard-classification pilot

Model: Qwen/Qwen3.6-27B-FP8, revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09. Local L40S inference. No hosted judge API.

Sample: 50 true positives and 50 false positives from each retained class-weighted probe's test predictions, seed 20261001, without replacement (300 total). Original population sizes and labels remain in sample_private.json. The model receives only judge_inputs.jsonl plus the frozen task rubrics. No scores, labels, outcome strata, or few-shot examples are disclosed.

Prompt design: common classifier instruction, task-specific positive/negative definitions, original messages serialized as JSON, and a final binary classification instruction. WildGuard targets the final assistant response using the request as context. ToxicChat targets user-message toxicity. High-Stakes targets situational consequences, not harmful-response generation. Rubrics are deliberate adaptations, not verbatim published prompts. Source text is treated as data, not operative instructions. Thinking is disabled. The model's next-token logits for the single-token labels 0 and 1 are compared; the higher scoring label is the hard decision (tie -> 0). No threshold fitting, free-form rationales, or retries of completed judgments.

Sources:
- https://raw.githubusercontent.com/allenai/wildguard/main/wildguard/utils.py — separates request harmfulness, response refusal, and response harmfulness.
- https://papers.nips.cc/paper_files/paper/2025/file/b9501301a7a0f99e57406cc901f4e3ce-Paper-Conference.pdf — Appendix A.2 compares constrained label prompts, and section 2.2 scores alternative continuations. Our one-token hard decision is an adaptation. The paper's cascade routes uncertain examples and averages scores; our positive-only confirmation pipeline is different.
- https://arxiv.org/pdf/2310.17389 — ToxicChat binary toxicity includes indirect requests to elicit toxic content; jailbreaking is separately annotated.
- https://huggingface.co/Qwen/Qwen3.6-27B-FP8 — official model and quantization.

Metrics: on the balanced flagged-only sample, probe-only TP=50, FP=50, FN=0, TN=0, so F1=2/3. Report judge/cascade results on that same sample. For full-test estimates let a be the fraction of sampled TP retained and b the fraction of sampled FP retained. Estimated cascade TP=original_TP*a, FP=original_FP*b, FN=original_FN+original_TP*(1-a), TN=original_TN+original_FP*(1-b). Compute F1 from these counts. These are estimates rather than a full-test judge evaluation; there is sampling uncertainty. No prompt tuning on these 300 test cases.

The inference script saves exact rendered prompts, model revision, input hash, label logits, hard decisions, timing, and durable progress. An exception is recorded, not converted into a negative class. Completion requires all 300 judgments.
