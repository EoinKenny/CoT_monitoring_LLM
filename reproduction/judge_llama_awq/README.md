# Llama AWQ recovery attempt

Authorized by the user's request to fix the failed Llama run on 2026-10-02.
Original attempt at /home/ubuntu/experiments/judge-llama-20261002 is preserved. Its RedHatAI compressed-tensors checkpoint failed configuration validation because vLLM 0.30 does not support actorder=group. No judgments were generated.

Replacement: ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4
Revision: a3c28ea93c041acfaadb85401ad6493749a5aa04
Quantization: AWQ INT4, group size 128, zero point enabled, awq_marlin backend.
This is a different quantization of the same Llama-3.3-70B-Instruct model; quantization provenance must be reported with results.

Remote root: /home/ubuntu/experiments/judge-llama-awq-20261002
Supervisor PID at launch: 348348
Baseline outputs: run/
Explanation outputs: saliency/run/ and nearest/run/
All 3 conditions contain the same 5,190 frozen flagged examples. Semantic prompt hashes are verified against the completed Gemma baseline and Qwen explanation prompts before inference. Greedy one-token hard classification, original rubric and probe thresholds preserved. Per-condition aggregate evaluations are written after completion.

No automatic retry after failure. Follow QUEUE_STATUS.json and per-condition run/status.json, FAILED.json and COMPLETED markers. Raw private data remains on the instance.

Live startup verification: AWQ compatibility passed, model loaded on L40S, and baseline hard-label judgments began successfully. Startup reported transient allocation failures during weight preparation but recovered without restart; raw baseline.log preserves these warnings. Semantic instruction checks passed for all three conditions. The first 44 judgments were confirmed produced; subsequent live validation checked unique IDs and binary outputs. Saliency and nearest-neighbour conditions remain queued behind baseline.
