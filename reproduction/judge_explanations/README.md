# Explanation conditions

Remote run: /home/ubuntu/experiments/judge-explanations-20261002

5190 identical flagged test cases x two explanation conditions x two models = 20760 new judgments. Existing no-explanation runs are the baseline. Qwen3.6-27B-FP8 and Gemma 4 31B Instruct QAT W4A16 use their pinned prior revisions, native templates, thinking disabled and greedy one-token 0/1 selection. Probe thresholds and rubrics remain fixed.

Saliency: segment each eligible original message with pysbd 0.3.4 English, clean=False, character offsets. For each sentence average the retained class-weighted probe's logits over overlapping eligible content tokens; choose the maximum, breaking ties by first occurrence. Padding, template markers and special tokens are excluded with the audited original pooling mask. Tool JSON escaping is mapped back to the original content. Verify pooled token logits against the trained pooled-example probe; maximum observed error 4.78e-7. Sentences are heuristic linguistic segments, not annotated ground-truth rationales.

Nearest neighbour: cosine similarity in pooled activations standardized by that probe's training-fitted scaler. Search the same dataset's training split only; exact exhaustive search; ties choose the smallest index. No selection by class or correctness. Show full original training messages, dataset label, and retained probe's prediction at its validation-tuned threshold. No test label or test probe score is shown.

Instruction text and test IDs were verified identical between the two models within each explanation condition. Maximum prompt lengths: Qwen saliency 4126, Qwen nearest 5921, Gemma saliency 3567, Gemma nearest 4482. No truncation.

Queue runs qwen_saliency, qwen_nearest, gemma_saliency, gemma_nearest sequentially on the GPU. Outputs are saved after each batch. Any failure stops the queue without retrying completed judgments. QUEUE_STATUS.json and per-condition run/status.json track progress. Aggregate evaluation runs after each completed condition. Detailed conversations and predictions remain on the remote instance.

Nearest-neighbour explanations include an extra labelled example; differences from sentence saliency therefore also reflect additional context and supervision, not only explanation format. Both conditions are the user-approved operational interventions, not a matched-information causal comparison.
