INTERPRETABLE PROBE: EXTERNAL API JUDGE HANDOFF

Give this directory to the agent that controls your API server. The agent should
run each selected API model on every row of prompts.csv and return responses.csv
plus model_manifest.json. No backbone extraction or probe training is needed.

SCOPE
There are exactly 66,000 prompts PER API MODEL:
  5 backbones x 10 datasets x 220 frozen cases x 3 explanation conditions
  x 2 output formats. Each backbone/dataset has 110 validation and 110 test cases.
The explanation conditions are none, saliency, and nearest training neighbour.
The output formats are hard labels and positive-class probabilities. Repeated
conversation text across rows is intentional: do not deduplicate or resample.
All target labels and evaluation weights remain with the originating project.
Training-neighbour labels in the supplied prompts are intentional and must stay.

INPUT: prompts.csv (UTF-8, standard quoted multiline CSV)
  prompt_id: immutable join key; never change it.
  output_type: hard or probability; metadata, not text to append to the prompt.
  system_prompt: send verbatim as the API system instruction.
  user_prompt: send verbatim as the API user message.

Send exactly these two messages in a fresh conversation for every row. Preserve
whitespace, Unicode, task rubrics, current conversation, and explanations. Do not
add instructions, assistant prefills, examples, metadata, gold labels or tools.
Do not summarize, truncate, rewrite, normalize or sanitize the prompt text.
Read the CSV programmatically (for example, Python csv.DictReader); spreadsheet
editors can corrupt long fields. Set csv.field_size_limit(100000000).

CALLS AND SETTINGS
Use the existing llm_query.py query function on this server if it supports these
exact messages and settings; inspect its actual signature first. Adapt the call
wrapper only. Do not invent a function signature or use a second LLM to clean up
responses. Execute at most 15 API requests concurrently, respecting rate limits.
Use temperature=0, top_p=1, fixed seed=20261001 when supported. Disable thinking /
reasoning where supported. If a model cannot disable reasoning, assign it a
separate model alias and record the limitation; do not label it non-reasoning.
Use a small but sufficient output budget (8 tokens for direct-output models is
the default; record any provider-specific alternative before the run). Preserve
provider finish reasons privately; a length-truncated response is invalid even
if the partial text looks numeric. Never silently trim inputs to fit context.

The local hard-label comparison uses constrained 0/1 decoding and the probability
comparison uses a constrained 0-100 integer. Use equivalent native scalar/regex
constraints only when supported without changing the supplied message text. Do
not wrap responses in JSON. Where constraints are unavailable, retain ordinary
text generation and document that decoding difference in model_manifest.json.
Keep all settings constant within a model alias and output type. API versions
and hosted model internals may change; record the resolved model version and
provider fingerprint when exposed, UTC run dates, and unsupported parameters.

RETURN: responses.csv, exactly these seven columns
  prompt_id,model,response,status,input_tokens,output_tokens,latency_ms

  model: stable alias identifying one exact configuration in model_manifest.json.
  response: the UNMODIFIED final response text. Do not convert it to a number,
    strip it, extract an integer from a sentence, or threshold probabilities.
  status: ok, invalid, error, ambiguous, or skipped.
    ok: after ignoring outer whitespace ONLY, hard matches [01], or probability
      matches (100|[1-9]?[0-9]); request finished normally.
    invalid: refusal, explanation, empty output, extra text, noninteger, value
      outside 0-100, or truncated generation. Keep the complete raw response.
    error: known failed request; put the concise provider error in response.
    ambiguous: timeout/disconnect for which generation may have occurred; put
      the diagnostic in response, and do not silently submit another request.
    skipped: intentionally unqueried row, with a short reason in response.
  input_tokens/output_tokens: actual provider usage counts; leave blank if not
    exposed. output_tokens must include billed hidden/reasoning tokens when the
    provider reports them (avoid double-counting). Document exact usage semantics
    and aggregate cached-input/reasoning breakdowns in model_manifest.json.
  latency_ms: per-request wall-clock elapsed time, including generation but
    excluding deliberate waiting before dispatch; blank only if unavailable.

Thus one response column serves either 0/1 labels or 0-100 probabilities. We join
prompt_id back to all dataset, backbone, explanation, split and evaluation data
here; you do NOT need to repeat those columns or return the prompt text.

RESUMPTION AND FAILURES
Persist a request-start record before dispatch and the raw provider response
immediately after it returns, using a private append-only request journal. Keep
request IDs, finish reasons, usage, settings, timestamps and failed attempts in
that journal. Do not put API keys in exported files. Save responses incrementally
and resume by (prompt_id, model). Never overwrite a completed response or create
duplicate keys. Do not automatically retry invalid or ambiguous generations.
Retries after a definite pre-generation rejection must be logged. If a later
run changes a model/version/settings, use a new alias; never merge silently.

COMPLETION AND RETURN
Start by verifying export_manifest.json's prompts_csv_sha256 and the ZIP contents.
Populate model_manifest.json from model_manifest.template.json. Store per-model
settings once there rather than adding columns to each response. Include all
selected aliases even if a model is incomplete. Run:
  python3 validate_results.py --responses responses.csv --manifest model_manifest.json
The validator makes no API calls. It reports status counts and missing coverage;
use --require-complete to enforce 66,000 valid responses per listed model.
Partial results are useful: return them with missing/errors honestly recorded.
Do not tune thresholds or compute headline F1 on this server. We will select
thresholds on frozen validation cases and evaluate held-out test cases here.
Return a ZIP containing responses.csv, model_manifest.json, validation_report.json
and any brief issues note. Preserve the larger raw request journal on your server
for audit; it need not be included in the transfer unless requested.

BUDGET REDUCTION (2026-10-11)
This package supersedes the earlier 300,000-prompt package. Use only these
66,000 rows per model. Prompt IDs and text are unchanged for retained cases.
If calls have already started on the older package, keep all those results,
reuse matching completed IDs, and stop dispatching IDs absent from this file.
Do not re-query a completed or ambiguous request. Return extra old results
separately; the validator for this package accepts only its selected IDs.
The originating project selected the same outcome-blind stratified subset for
all judges and conditions and will apply revised sampling weights there.
