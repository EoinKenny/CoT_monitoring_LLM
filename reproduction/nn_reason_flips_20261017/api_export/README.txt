API 1-NN FLIPS — PROBABILITY FIRST, VERY BRIEF EXPLANATION AFTER

This is a NEW follow-up study for the four prior aliases:
claude4.5-sonnet, claude4.8-opus, gpt-4o, gpt-astra.
Use the same backend models as the earlier run if still available. These aliases
are not verified backend model IDs. Resolve them from your previous configuration
and record exact provider/model/version details in model_manifest.json. If a
backend changed or is unavailable, report that rather than silently substituting.

WHAT TO RUN
Read prompts.csv with Python csv.DictReader. It contains quoted multiline text.
Each unique prompt appears ONCE. The models column is a JSON array of the aliases
that should receive that row. Run ONLY those (prompt_id, model) pairs, NOT every
row on all four models. See export_manifest.json for exact counts.

Each model receives its OWN prior test cases where the no-explanation and 1-NN
answers disagreed at the common probability cutoff 50. Both helpful and harmful
flips are included. No validation cases, control calls, saliency, or 2-NN calls.
Invalid/missing original comparisons and ambiguous request histories are excluded
with counts in export_manifest.json. Different models therefore have different
case sets. Do not resample or deduplicate cases beyond the provided routing.

HOW TO CALL
Import your existing query function from llm_query.py in the API server's existing
directory/environment. Adapt its call signature there; do not replace the module.
Use at most 15 concurrent API requests. One fresh conversation per routed pair:
  [{"role":"system","content":row["system_prompt"]},
   {"role":"user","content":row["user_prompt"]}]
Send these exact strings. Do not add model routing/IDs, instructions, examples,
tools, original answers, or hidden labels. Training-neighbour labels and probe
predictions already inside the user prompt are intentional; preserve them.
The original API task/reference/target text is unchanged. Only the output
instruction changes to request the probability first, then a VERY brief explanation.

SETTINGS
Temperature 0; seed 20261001 where supported; max final output tokens 256.
Disable native thinking/reasoning where supported. Do not raise the token cap or
introduce retries/temperature changes for repetitive output. Native text/regex
constraints may enforce integer-newline-text if available; no JSON wrapper.
Record actual parameters, provider restrictions, model snapshots/fingerprints,
reasoning-token accounting, and any provider-required budgets in the manifest or
request journal. If a provider cannot meet the requested settings, document the
deviation; never silently treat a changed backend/configuration as identical.

OUTPUT CONTRACT
First line: one integer 0..100 estimating POSITIVE-CLASS probability.
Next line: VERY briefly explain the reasoning behind the assessment.
For example (format only, not added to the prompt):
85
The target shares the relevant harmful intent with the labelled neighbour.

SAVE responses.csv WITH EXACTLY SIX COLUMNS
prompt_id,model,probability,raw_response,status,finish_reason

raw_response: UNMODIFIED final visible response, including whitespace, later
numbers, contradictions or repetition. Save raw provider payloads separately in
the journal when needed, including refusals and any separate reasoning fields.
probability: parse ONLY the first line of raw_response.strip(); after trimming that
line it must fully match (?:0|[1-9][0-9]?|100). Otherwise leave probability blank.
Never search later text for a number. Never replace the first number with a later
one, even if the explanation contradicts it. Do not strip a thinking preamble to
manufacture a valid probability. Keep native private reasoning separate from the
final visible response if the API provides distinct fields.
status:
  ok        valid first probability, even if explanation is blank or cut off
  invalid   final response exists but first line is not a valid integer
  error     request definitely failed; preserve error details, probability blank
  ambiguous timeout/disconnect where generation may have occurred; blank probability
  skipped   explicitly never submitted; give reason in journal, blank probability
finish_reason: provider's actual value, e.g. stop/length/content_filter; blank if
unavailable. Preserve truncation as data. It does not invalidate a valid first
probability. Missing or malformed explanations are not grounds for another call.

RECOVERY AND JOURNAL
Write and flush a durable request_journal.jsonl dispatch record BEFORE each call,
keyed by (prompt_id,model), including UTC time, prompt checksum and config ID.
Immediately save the result/raw text, request ID, finish reason, error/status,
latency and input/output/reasoning token counts if supplied. Persist response rows
durably. On resume, submit only never-dispatched pairs. Disable automatic SDK
retries. Never silently repeat or overwrite completed, invalid, or ambiguous calls.
Report unresolved dispatched pairs explicitly as ambiguous; retain all attempts.
Token and timing metadata belong in the journal, keeping responses.csv minimal.

VALIDATE AND RETURN
Before calling APIs:
  python validate_results.py
After collection (partial results also accepted):
  python validate_results.py --responses responses.csv --manifest model_manifest.json
When every routed pair has an outcome (including explicit invalid/missing outcomes):
  python validate_results.py --responses responses.csv --manifest model_manifest.json --require-accounted

Return a ZIP with responses.csv (or.csv.gz), model_manifest.json,
validation_report.json and request_journal.jsonl (gzip allowed), plus relevant raw
provider/error sidecars. Do not include API keys or authentication headers.
Do not return the input prompts again. Do not tune thresholds on this selected
test subset. The original agent retains all gold labels, design weights, original
scores and helpful/harmful mappings privately for the paired follow-up analysis.
