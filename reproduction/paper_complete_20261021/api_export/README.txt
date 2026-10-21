INTERPRETABLE PROBE — COMPLETE SIX-API PAPER MATRIX (2026-10-21)

Read this before running. This is an implementable handoff for the agent on the
API server. The user authorized this collection, including both explanation-order
experiments. Implement the provider adapter on your server; the supplied kit is
offline and makes no API calls. Preserve the exact messages and all failed outcomes.

JUDGES
Existing aliases: claude4.5-sonnet, claude4.8-opus, gpt-4o, gpt-astra.
New requested aliases: claude-opus-5.5, gpt-5.5.
Resolve exact provider IDs, snapshots and inference settings using your existing
configuration, account availability and current provider documentation. Do not
invent a model ID, silently substitute a model, or map two names to the same model.
For the four existing aliases, reproduce their original configuration and record
the prior manifest hash. A changed backend/version is a new configuration, not a
matched continuation; report that before running the affected followups. If a
requested model is unavailable, record it as blocked and continue other models.
For both new judges, freeze the configuration before their main data collection.

WHAT TO COLLECT
Each of six judges:
  ablation     350,000 = 5 backbones x 10 datasets x 1,000 cases x 7 conditions
  two_nn        50,000 = same cases, two activation neighbours
  mechanism     25,000 = all original test contexts; similarity/contrast JSON
New judges only:
  main         150,000 = control, saliency, one activation NN on the same cases
Existing four judges additionally:
  reason_after + reason_first: 7,122 calls per order across the four judges,
  selected separately from each model's original valid, unambiguous test flips.
New judges additionally:
  both trace orders on their OWN test flips, selected after main completes.

Static queue: 2,864,244 model-prompt calls, less any already attempted old exports.
The two new judges add 0..100,000 trace calls combined, depending on their flips.
No semantic-retrieval, fresh-sample or constructed-pilot expansion is requested.
All NNs here use the original backbone activation retrieval. No retraining.
Old main responses are already held by the originating researcher; do NOT rerun
main for the original four models. Import your existing journals before starting:
if a 2NN or probability-first trace export has already been attempted, reuse it.
The old probability-first IDs and texts are preserved exactly. Both reasoning
orders are independent conversations on the same selected original flip cases.
Do not select reasoning-first cases based on the probability-first answers.

FILES AND EXACT PROMPTS
prompts.sqlite is a lossless, read-only catalog. Repeated text fragments are stored
once; kit.py reconstructs exact roles/messages and verifies each message hash.
It contains 625,000 prompt records, including unassigned trace templates for the
two new models. Only tasks returned by iter_jobs are authorized for submission;
do not submit all catalog rows to every judge.
Metadata, IDs, study names and split names must NEVER be added to model messages.
Target gold labels, original judge scores and helpful/harmful directions are NOT
in this package. Reference training labels are intentional prompt content.
Mechanism prompts intentionally contain the original target probe score, as in
the approved annotation study; do not add it to classification prompts.
Each request is a new conversation. No browsing, tools, conversation history,
prompt rewriting, text truncation, label substitution or case resampling.

QUICK START (Python standard library only)
  python kit.py verify
  python kit.py export-csv --model gpt-5.5 --study main --out gpt55_main.csv
  python kit.py export-csv --model claude4.8-opus --study ablation --out opus_ablation.csv

CSV columns: prompt_id,model,study,messages_json,max_output_tokens.
messages_json is a JSON list of exact role/content objects. Prefer streaming to
avoid expanding huge CSVs:
  from kit import iter_jobs
  for job in iter_jobs('gpt-5.5', 'main'):
      # submit job['messages'] unchanged using your provider adapter
      ...

RECOMMENDED ORDER
1. Verify hashes, inspect supplied unit tests, freeze model manifests, import all
   previous attempt/result journals. Create a durable unique ledger keyed by
   (model alias, prompt_id); its UNIQUE constraint must prevent concurrent duplicates.
2. For each model/study, use the first 32 scheduled tasks (two batches of16) as the
   integration smoke. Retain them in the full dataset, never send them again.
   Check role mapping, output budget, actual model identity and durable raw saves.
   Model refusals/invalid outputs are observations, not a reason to retry or skip
   the corresponding dataset. Fix transport/configuration bugs before scaling.
3. Run static queues for all six judges. Run main for the two NEW judges first
   when convenient, so their trace queues can be constructed promptly.
4. Select new-model trace cases from their complete main outputs and journal:
   python kit.py select-traces --model gpt-5.5 --responses responses.csv \
       --journal request_journal.jsonl --out gpt55_trace_selection.json
   python kit.py export-csv --model gpt-5.5 --study reason_after \
       --selection gpt55_trace_selection.json --out gpt55_reason_after.csv
   python kit.py export-csv --model gpt-5.5 --study reason_first \
       --selection gpt55_trace_selection.json --out gpt55_reason_first.csv
   Repeat for claude-opus-5.5. Preserve these selection receipts.
   All150000 main outcomes must be accounted for, including failures. Only valid
   single-attempt baseline/NN pairs enter flip selection. Select when their >=50
   classifications differ. No labels, outcomes, or test-performance optimisation.
5. Validate, reconcile expected/attempted/returned/invalid/ambiguous/skipped counts
   per model/study, then return the artifacts listed below. Do not claim complete
   while any never-submitted task lacks an explicit disposition.

GENERATION SETTINGS AND PARSING
Temperature0, seed20261001 where supported, native/hidden thinking disabled where
supported. Record unsupported settings and actual values explicitly. Do not raise
temperature to repair outputs. Do not change budgets mid-run.
  main/ablation/two_nn: final output cap8; one integer0..100 only.
  mechanism: cap128; JSON with exactly labelled_similarity and important_contrast,
             each yes/no/uncertain. No evidence or explanation requested.
  reason_after: cap256; first standalone line is probability, followed by a VERY
                brief explanation. Score the first number even if prose later
                contradicts it or the explanation reaches the cap. Retain cap flag.
  reason_first: cap512; VERY brief explanation then a final standalone integer.
                Require complete normal finish. Capped generations stay unscored,
                even if the last partial text happens to look like a number.
These are expressed explanations, not access to private chain-of-thought.
Native hidden reasoning must not be enabled just because the task is named reason.
Use kit.parse(study, raw_final_text, normalized_finish_reason). Preserve raw text.
Normal finish reasons accepted: stop,end_turn,completed,eos,stop_sequence.
Map provider-native completion statuses transparently; keep originals in raw files.
For main outputs, scalar constraints are allowed where supported. Never replace
scalar output with JSON. Mechanism generation was unconstrained in the local run;
use the approved prompt, without adding a new format/schema instruction.

OPENAI / CLAUDE ADAPTER NOTES
Official GPT-5.5 documentation lists reasoning.effort='none' as supported; set it
explicitly for this experiment because the model defaults to medium. Verify the
endpoint's supported temperature, seed and token-budget parameters rather than
blindly passing generic names. Source checked2026-10-21:
https://developers.openai.com/api/docs/models/gpt-5.5
For Claude Opus5.5, verify the exact available provider ID and supported thinking
configuration on your server. Do not assume that label is a public snapshot ID.
If an endpoint cannot meet the visible output budget or disable native thinking,
report the incompatibility and proposed configuration; do not silently broaden the
budget or substitute a model. Continue independent model/study work.
Use your previously approved provider interface. Batch/caching is acceptable if
it preserves independent exact messages, model/settings and stable request IDs.
Keep reasonable concurrency (default at most15 outstanding requests) and provider
rate limits. Disable automatic SDK retries for ambiguous dispatched requests.

DURABLE RECOVERY
Write/fsync a dispatch event BEFORE submission, including model, prompt_id,
study, messages_sha256, configuration hash, attempt ID and timestamp. Persist raw
provider response BEFORE parsing, then write/fsync result and final CSV row.
If transport disconnects after submission, do not blindly retry. Recover by the
same provider request/batch ID if possible; otherwise retain ambiguous outcome.
Likewise never retry refusals, invalid, truncated, or already completed outputs.
Only never-dispatched tasks may be submitted on resume. Preserve all attempts,
including errors. Do not repeatedly rerun a smoke after poor model answers.

MINIMAL RESPONSES CSV — EXACTLY EIGHT COLUMNS
prompt_id,model,response,status,finish_reason,input_tokens,output_tokens,latency_ms
response: untouched final visible text (proper CSV quoting preserves newlines).
status: ok,invalid,error,ambiguous,skipped. Error details belong in the journal/raw
sidecar. Blank raw output remains blank; do not invent a probability or class.
Keep provider request IDs, backend fingerprint, cached/reasoning-token accounting,
configuration, exceptions and timestamps in the durable journal/raw sidecars.
For each model-prompt pair, at most one canonical CSV row. Preserve duplicate old
attempts in the journal; mark ambiguity instead of selecting the best response.

JOURNAL MINIMUM SCHEMA
One JSON object per line. Both dispatch and result include:
  event,model,prompt_id,attempt_id,time_utc,configuration_sha256
A dispatch also includes study,messages_sha256 and actual request parameters.
A result includes status,finish_reason,raw_path,raw_sha256,provider_request_id.
Record all provider metadata and usage available. Metadata must never contain keys.
Use event='dispatch' and event='result' so the supplied selector can check single
unambiguous pairs. For recovered results retain the original dispatch/attempt ID.

RETURN AS ONE ZIP
responses.csv (or partitioned per-model CSVs with exactly the same columns)
request_journal.jsonl; model_manifest.json; validation_report.json
raw provider response files; new-model trace_selection.json receipts
package checksum; implementation/source/settings needed to reproduce the run.
Model manifest must resolve all aliases and preserve settings per study.
Run: python kit.py validate --responses responses.csv --journal request_journal.jsonl
Also produce a full expected-versus-accounted coverage report from iter_jobs,
including new dynamic queues. The validator reports parsed outcomes; its presence
alone is not a completeness assertion. Do not execute further analysis or adjust
thresholds on test outcomes; the originating project performs the paired analysis.
