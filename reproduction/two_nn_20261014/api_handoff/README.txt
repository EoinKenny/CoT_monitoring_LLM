INTERPRETABLE PROBE — PROBABILITY ONLY, 500 VALIDATION + 500 TEST

Run each selected external API model on every row of prompts.csv using the
existing llm_query.py query function on your API server. Use at most 15 concurrent
API calls. This package adds the two-neighbour condition. Do not submit the prior control,
saliency, or one-neighbour prompts again.

SCOPE
50,000 prompts PER API MODEL = 5 backbones x 10 datasets x 1,000 cases x 1
new explanation condition: two nearest training neighbours (nearest2).
Run the same four API judge configurations as the original comparison.
Keep all prior responses; this adds one condition and does not replace them.
Each backbone/dataset has 500 validation and 500 test cases, sampled across the
probe-score range using the original frozen stratified design, not top500 scores.
Use every row. Do not resample, deduplicate, truncate or change explanations.
Target gold labels and sampling weights remain private on the originating server.
Training-neighbour labels in prompts are intentional and must remain intact.

INPUT
prompts.csv is UTF-8 with quoted multiline fields. Use a proper CSV reader.
Columns: prompt_id,output_type,system_prompt,user_prompt.
For each row, send the exact system and user messages in a fresh conversation.
Do not add the metadata columns to the prompt, extra examples, or tools.

OUTPUT
Exactly one integer 0..100, e.g. 85. No label, comma, JSON, percent sign or prose.
The prompt explicitly defines >=50 as positive and <50 as negative, with 100
certainly positive and 0 certainly negative. The score is positive-class
probability, not confidence in whichever class the model favors.
Allowed regex: (?:0|[1-9][0-9]?|100).
Do not parse, threshold or reformat the response before saving the raw text.

SETTINGS
Temperature=0, seed=20261001 where supported, non-reasoning where supported,
max final output tokens=8. Use native scalar/regex constraints if supported;
otherwise plain text. Do not introduce JSON-format constraints.
Record exact model/provider IDs, versions/fingerprints, settings, run dates,
prompt checksum, unsupported settings and reasoning/token accounting in
model_manifest.json using the supplied template. If a provider requires internal
reasoning or another budget, record it explicitly and keep its configuration fixed.

RESULTS — SAVE responses.csv WITH EXACTLY THESE SEVEN COLUMNS
prompt_id,model,response,status,input_tokens,output_tokens,latency_ms
model: stable alias for one exact configuration in model_manifest.json.
response: UNMODIFIED raw final text, stored as text, or error diagnostic.
status: ok, invalid, error, ambiguous, skipped.
  ok: normal completion, full regex match after trimming outer whitespace only.
  invalid: refusal, extra text, empty/truncated output, noninteger, out-of-range.
  error: known failed request; preserve diagnostic.
  ambiguous: timeout/disconnect where inference may have occurred; do not retry.
  skipped: explicitly unsubmitted, with reason.
Tokens: provider counts, blank if unavailable. Latency: request elapsed ms.
Keep reasoning-token details in a sidecar/manifest if needed, not extra CSV columns.

RECOVERY
Durably journal attempts before submission, keyed by (prompt_id,model); save
responses immediately. Resume only never-submitted rows. Do not silently retry
ambiguous or invalid attempts, replace responses, or duplicate completed keys.
Preserve provider traces separately where needed. Report failures without hiding them.

VALIDATE AND RETURN
Before running: python validate_results.py --prompts prompts.csv
After running: python validate_results.py --prompts prompts.csv \
  --responses responses.csv --manifest model_manifest.json
Add --require-complete when all expected responses should be valid.
Return responses.csv (gzip allowed), model_manifest.json, validation_report.json
and any failure/ambiguous-attempt sidecar. Partial results are acceptable.
The originating agent joins by prompt_id and evaluates fixed50 and validation-
tuned cutoffs with matched cases and design weights. Do not tune thresholds here.
