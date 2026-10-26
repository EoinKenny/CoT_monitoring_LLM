# Rebuild the open-source experiments

```bash
python3.12 reproduce_open_source.py
```

Run from the cloned repository on Linux with Python 3.12, a working NVIDIA
driver, access to the pinned model repositories, and enough storage. The full
configuration targets a 96 GiB GPU (tested hardware: RTX PRO 6000 Blackwell),
at least 128 GiB host RAM and approximately 1 TB free disk. Llama 70B offloads
part of its exact BF16 prefix to CPU. Other GPU architectures may not support
the original NVFP4/FP8 kernels. Smaller selections need less memory/storage.

Accept the Llama/Gemma model licences on Hugging Face and provide `HF_TOKEN`
through the environment or your existing Hugging Face login. While this GitHub
repository is private, dataset assets also require `gh auth login`, `GH_TOKEN`
or `GITHUB_TOKEN`. Credentials are not written into experiment records.
Python must have `venv`/`ensurepip`; on Debian/Ubuntu this may require the
`python3.12-venv` package. CUDA compilation support and a driver compatible with
the pinned PyTorch CUDA builds are required.

## What is regenerated?

1. Download the checksummed, prepared **original dataset records and fixed
   train/validation/test memberships**, including native tokenization where it
   was saved. This 2.04 GiB input supplement contains no activations, fitted
   probes, probabilities, retrieved neighbours or judge responses.
2. Download pinned backbone revisions; extract new content-token pooled
   activations at the original layers. Large backbones execute only the causal
   prefix required for the probe, before final normalization.
3. Fit a train-only `StandardScaler` and a new balanced-weight logistic probe
   for every backbone/dataset. Use the original C, seed, solver and validation
   F1 rule. There is no backbone fine-tuning.
4. Recreate the validation-defined score-decile × gold-label stratified samples
   (500 validation and 500 test records per cell), with original sampling seed
   and inverse-inclusion weights. The newly fitted probabilities determine the
   sample; no historical judge result influences it.
5. Retrieve the exact two closest distinct training examples by cosine in the
   standardized backbone activation space. Ties use immutable training IDs.
   Recompute sentence saliency from fresh token activations. Reference probe
   predictions use the newly fitted probe's validation-selected threshold.
6. Run six local judges with pinned revisions, original quantization/template
   settings, native thinking disabled, temperature 0, seed 20261001, and the
   approved prompts. Save raw responses, token IDs, finish reasons and failures.
7. Run both reasoning orders on each judge's **own newly observed** control/1-NN
   test flips at threshold 50. Probability-first has a 256-token cap;
   explanation-first has 512. Score the first probability for probability-first
   even if later text disagrees. Truncation stays recorded.
8. Qwen122, Qwen27B and Gemma label similarity/contrast in the 1-NN test contexts;
   Qwen122 labels explicitly expressed comparisons in the probability-first
   explanations using the final user-approved three-key prompt.

The command rebuilds probes and neighbours **from dataset records**, not from
the saved probe outputs. Dataset acquisition, cleaning, deduplication and split
creation are frozen at their audited historical state, rather than re-scraping
moving upstream datasets. Those historical adapters remain in
`remote_collection/`. Model weights are downloaded from their providers, not
redistributed in the dataset supplement; original dataset terms still apply.

This is a fresh collection under the study protocol. Hardware, kernels,
batching and runtime differences can change probabilities, neighbours, selected
rows and flips. Bitwise identity with historical results is not promised.
Numerical saliency replay differences are reported as diagnostics, not used as
agreement gates. Use `python reproduce_results.py` for the published figures
from frozen data, or `--recompute` to rebuild their statistics.

## Useful options

```bash
# Print the entire plan without downloading or running models.
python3.12 reproduce_open_source.py --dry-run

# Small real end-to-end smoke; never use its statistics as paper results.
python3.12 reproduce_open_source.py --smoke --backbones llama --judges gemma --trace-judge gemma --datasets toxicchat --out outputs/smoke

# Only the four main explanation conditions, for a subset of the grid.
python3.12 reproduce_open_source.py --backbones llama,qwen --judges qwen,gemma --datasets toxicchat,toxigen --studies main,two_nn --out outputs/subset
```

`--stop-after inputs|backbones|prompts|judges` prepares a partial run; repeat the
same arguments without that option to continue. `--model-cache /large/disk/hub`
changes the weight-cache location. `--inputs-dir` reuses verified input assets.
`--model-paths paths.json` can supply existing pinned snapshot paths under
`{"backbone":{"llama":"..."},"judge":{"gemma":"..."}}`; explicit overrides
are recorded and must refer to the documented revisions. Nothing is uploaded
from a collection run.

By default environments are created inside the output directory. Advanced users
can supply `--backbone-python`, `--qwen-backbone-python`, `--judge-python` and
`--mistral-overlay` for compatible existing runtimes. Qwen122 FP8 extraction
uses its original Transformers 5.18/PyTorch 2.13 runtime; the four other
backbones use Transformers 5.9/PyTorch 2.9. These environments are isolated.
Mistral alone needs the original Transformers
5.4/tokenizers 0.22.2 overlay; other judges use 5.18/0.23.2. The command installs
this overlay without changing the shared judge environment.

## Outputs and resumption

Everything goes under `outputs/open_source_rebuild` (or `--out`):

- `RUN.json`, `PLAN.json`, `STATUS.json`: immutable settings/code hashes and
  current progress.
- `backbones/`: new activation features, fitted probes, full probabilities,
  sampled cases, population counts, 1-NN/2-NN/random references and saliency.
- `prompts.sqlite`: exact message contents and private case/sampling metadata.
- `responses.sqlite`, `judges/*/batches/`: durable dispatch journal, raw outputs,
  input/output token IDs, provenance and explicit ambiguous/failed attempts.
- `exports/responses.csv.gz`, `cases.csv.gz`, `scores.csv.gz`,
  `trace_sources.csv.gz`, `coverage.json`: analysis-ready tables and joins.
- `OUTPUT_SHA256.json`, `COMPLETED.json`: checksums and final counts. Completion
  means all scheduled calls are accounted for, not that every response is valid.

A controller lock prevents duplicate launches into the same output directory.
Committed stages are hash checked. A response is recorded as dispatched before
inference; a crash with no durable raw response leaves that call ambiguous and
it is **not resubmitted**. A saved raw response can be parsed on resume without
inference. Invalid responses remain invalid. Resume only runs unattempted calls.
Keep the same code and arguments when resuming; changed inputs/settings require
a new output directory. A 20 GiB free-space reserve is enforced. Extraction
chunks can be recomputed if interrupted before commit because they contain no
judge responses; committed extraction chunks are verified and reused.

The new exports are independent data products: they do not overwrite the
published scores or automatically substitute into the historical figure bundle.
The paper's 50/50 and four-anchor validation-controlled evaluation code remains
in `paper/` and the original experiment sources.

## Validation

All five backbones passed a 48-record ToxicChat GPU smoke, including fresh
probe fitting, neighbours and saliency. The Llama/Gemma end-to-end smoke
produced 92 valid responses and resumed without repeating inference. All 50
backbone/dataset tokenization checks and 16 unit tests passed. The other five
judge engines and the reasoning/trace branches were not exercised end-to-end
in this small test; a complete rerun was not launched. See the
[validation receipt](OPEN_SOURCE_VALIDATION.json) for exact scope.
