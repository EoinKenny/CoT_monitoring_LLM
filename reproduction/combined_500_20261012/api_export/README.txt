INTERPRETABLE PROBE — COMBINED LABEL + PROBABILITY, 500+500

Give this package to the agent on your API server. Run each API model you choose
on all 150,000 rows of prompts.csv. This replaces the earlier 60,000-row export.

SCOPE
5 backbones x 10 datasets x (500 validation + 500 test) x 3 explanation
conditions = 150,000 calls PER API MODEL. Both outputs come from one response.
Conditions: no explanation, saliency, nearest training neighbour. Target gold
labels and evaluation weights are kept privately by the originating project.
Labels on training-neighbour examples are intentional: preserve them.
These are the original frozen full-threshold-sweep cases; preserve their IDs.
No new training, embeddings, sampling, or threshold tuning is required here.

PROMPTS
Read prompts.csv with a proper UTF-8 CSV reader; fields can contain quoted
newlines and commas. Columns: prompt_id,output_type,system_prompt,user_prompt.
Send the exact system and user messages as a fresh conversation for each row.
Do not include prompt_id or output_type in the message. Do not edit text,
truncate, combine rows, deduplicate repeated content, add examples or tools.

OUTPUT CONTRACT
Return exactly label,probability, for example 1,85 or 0,20.
Label is 0 or 1. Probability is an integer 0..100 estimating the POSITIVE class,
not confidence in the chosen label. Do not enforce agreement between label and
probability or convert one into the other. Do not ask for JSON or reasoning.
The exact allowed regex is [01],(?:0|[1-9][0-9]?|100).
This is a new prompt contract: previous separate-format results cannot be reused.

API RUNNER
Use your existing llm_query.py module in its directory; import its query function
using that module's actual signature. Keep at most 15 API calls in flight.
Use temperature=0, seed=20261001 where supported, non-reasoning mode where
supported, and a final-output budget of 16 tokens. Keep settings fixed within a
model alias. If a provider requires internal reasoning or a different budget,
record those exact settings and separate reasoning-token counts in the model
manifest/provider sidecar. Do not silently alter prompts to accommodate APIs.
Use a native regex/text constraint if supported; otherwise request plain text
and validate it. Do not add a JSON schema that changes the response format.
Record provider/model ID, version/fingerprint, settings, unsupported settings,
run dates and prompt-file checksum in model_manifest.json (template supplied).

SAVE responses.csv WITH EXACTLY THESE SEVEN COLUMNS
prompt_id,model,response,status,input_tokens,output_tokens,latency_ms

model: stable alias for a single exact configuration in model_manifest.json.
response: unmodified raw final response, kept as a string (e.g. "1,85" in CSV).
Use a CSV writer so the comma inside responses is correctly quoted.
status: ok, invalid, error, ambiguous, or skipped.
  ok: normal completion and full regex match after trimming outer whitespace.
  invalid: malformed, refusal, truncated, empty or out-of-range output; retain it.
  error: known failed request; retain diagnostic in response.
  ambiguous: timeout/disconnection where generation may have occurred; retain
    diagnostic and do not silently resubmit.
  skipped: explicitly unsubmitted row with reason.
input_tokens/output_tokens: provider usage, or blank if unavailable. Record
reasoning-token accounting semantics in the model manifest if applicable.
latency_ms: elapsed request time, or blank if unavailable.

RESUMABILITY
Before submitting, durably record an attempt keyed by (prompt_id,model).
Persist each response immediately. Resume only rows never submitted; never
silently retry unresolved attempts or duplicate completed rows. Preserve a
provider trace/attempt sidecar if needed, but keep responses.csv minimal.
Do not retry invalid outputs to obtain a different answer. Report failures.

VALIDATION AND RETURN
Before running: python validate_results.py --prompts prompts.csv
After running: python validate_results.py --prompts prompts.csv \
  --responses responses.csv --manifest model_manifest.json
Use --require-complete only when expecting complete valid coverage.
Return responses.csv (gzip is fine), model_manifest.json, validation_report.json,
and any failed/ambiguous-attempt sidecar. Partial results are acceptable; retain
all statuses. No need to send the large prompt CSV back.

The originating agent joins on prompt_id, parses both values, tunes thresholds
on validation only and performs matched weighted test comparisons. Do not
calculate or select F1 thresholds on this API server.
