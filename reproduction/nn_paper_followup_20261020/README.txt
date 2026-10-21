FOCUSED PAPER FOLLOW-UP — 20 OCTOBER 2026

Remote root: /home/ubuntu/experiments/nn-paper-followup-20261020
Local source: this directory. PROTOCOL.json fixes the study before collection.

1. k-NN versus linear probe: two backbones (Qwen, Llama70), seven datasets,
   500 existing validation and 500 test cases per cell, with design weights.
   Train/reference pool is the original training data. Cosine and Euclidean,
   k=1,5,15,51; select k/metric/cutoff using validation only. No new LLM calls.

2. Matched retrieval: 100 validation and 100 test cases per cell. Two newly
   collected arms use identical neutral wording and identical information
   fields; only the training reference selected by activation-space versus
   generic semantic similarity changes. Three local judges: Qwen122, Gemma,
   GLM. Original no-explanation and NN generations are retained separately.
   16,800 new judgments. Frozen neighbour prediction threshold preserved.

3. Fresh-case confirmation: up to100 validation and100 test cases per cell
   after excluding groups from prior selected_flags manifests. Three arms:
   no reference, activation reference, semantic reference. Up to25,200 calls.
   Exact eligible counts/exclusion files are saved; exhausted cells remain
   explicit. This is fresh relative to the audited selection manifests, not
   a new dataset or distribution and not a guarantee about unindexed pilots.

4. Controlled diagnostic pilot: 32 assistant-authored targets, five arms,
   three judges =480 calls. Endorsement/condemnation and possible/certain
   factual claims. Same-class, opposite-class, irrelevant and wrong-label
   references plus baseline. No invented probe prediction is supplied.
   Lexical variants are not independent mechanism replications. These are
   synthetic diagnostic observations awaiting independent human validation,
   not natural benchmark ground truth or a completed causal mechanism study.

All answers are probability-only0..100, greedy, max8 output tokens, thinking
mode disabled. Raw generations and attempted batches persist. No automatic
retry of ambiguous calls. Existing experiments are untouched. Reserve20GiB.

Human review:
Natural preparation creates human_review/packet.jsonl with200 outcome-blinded
reference prompts and blank review fields. Two independent human readers
should assess applicable analogy, misleading analogy and target ambiguity.
No human annotations have yet been collected. Controlled target/reference
labels require independent checking too. Do not describe assistant checking
as independent human validation.

Launch:
LAUNCH_GUARD prevents duplicate launch. LAUNCH.json holds process IDs. Read
STATUS.json, knn/STATUS.json and runs/<phase>/<judge>/STATUS.json. SMOKE.json
contains measured two-batch speed; it is not a reliable whole-study ETA until
natural prompts are prepared and measured. Never restart a controller based
only on an absent process: inspect all attempted and committed outputs first.
