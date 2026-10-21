Seven-judge expansion of the frozen probe -> LLM study
Started: 2026-10-11
Server root: /home/ubuntu/experiments/judge-expansion-20261011

Qwen27B and Gemma31B retain their existing, verified 600,000 combined outputs.
The controller collects MistralSmall4, OLMo32B, GLM4.7Flash, Qwen122B and the
reinstated Llama70B sequentially. Each adds 300,000 judgments: five backbones,
ten datasets, 500 validation + 500 test cases, three explanation conditions,
and two output formats. Total completed grid will contain2,100,000 judgments.

No backbone extraction, probe retraining, resampling, or explanation changes.
PROTOCOL.json and judge_specs.json freeze the exact scope and model settings.
canonical/cases.jsonl contains the shared task messages, with a checksummed
mapping to source prompts. Target gold labels never appear as target labels
in judge messages. Training-neighbor labels remain as intentional explanations.

Per-model prior proven batch sizes: Mistral16, OLMo8, GLM8, Qwen1224, Llama4.
Non-reasoning, greedy seed20261001; one-token hard label or an integer0-100
positive-class probability constrained with an8-token maximum. Tokenization
and maximum model context are computed from every full prompt; no truncation.
Mistral uses authoritative tokenize=True template output directly and an
isolated Transformers5.4/tokenizers0.22.2 overlay. Other judges keep their prior
runtime. All versions, model revisions and precise inference settings are saved.

Runtime and recovery:
 - Controller is detached and holds a lock; GPU judges execute sequentially.
 - Each batch saves an intent, raw outputs, then an atomic checksummed commit.
 - A completed batch is validated against its raw output before reuse.
 - Ambiguous attempts or invalid outputs stop safely for investigation; there
   is no silent retry, regeneration or dropping of cases.
 - Project and model-cache volumes preserve a20GiB free-space reserve.
 - Prior artifacts and unrelated keepalives remain untouched.
 - Numerical agreement/replay gates removed by the user remain removed.

Inspect STATUS.json, CHILD.json, PREPARE_STATUS.json and models/<judge>/STATUS.json.
Per-cell artifacts use hard/<backbone>/judges/<judge>_<condition>/run and the
corresponding probability/ path. Existing completed Qwen/Gemma cells are
read-only references to the original runs; expanded reports are separate.

After all collection is complete, evaluation/evaluate_expansion.py verifies
all2.1million outputs and runs the unchanged historical numerical estimators:
validation-only gate/cutoff selection, weighted held-out test estimates,
paired comparisons, calibration and stability diagnostics. Adds weighted
input-token costs at each selected gate. The controller runs this automatically.
Check analysis/report.html and root COMPLETED.json for final completion.

API handoff:
api_export/external_api_judges_20261011.zip contains300,000 exact prompts for
each external model, the agent README, minimal response CSV header, a per-model
configuration manifest template, and a no-API validator. The returned response
CSV uses seven columns: prompt_id,model,response,status,input_tokens,output_tokens,
latency_ms. All other scientific metadata join locally through prompt_id.
No API calls are performed by this local run.

Checks before launch:24 runner tests and7 evaluation tests passed, including
invalid/truncated output handling, tamper rejection, exact resume reconstruction,
threshold tie rules, weighted estimators and input-token-cost accounting.
