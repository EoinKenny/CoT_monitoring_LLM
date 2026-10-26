# Prompts and raw responses

The [raw-data supplement](https://github.com/EoinKenny/CoT_monitoring_XAI/releases/tag/raw-data-2026-10-26)
contains saved records for the paper experiments. It is optional: the small
`data/paper/` inputs already reproduce all plotted statistics. Release assets
keep large raw records out of normal Git clones.

```bash
gh release download raw-data-2026-10-26 --repo EoinKenny/CoT_monitoring_XAI --dir data/raw
python scripts/verify_raw_supplement.py data/raw --deep
```

The download inherits the repository's access permissions. Each ZIP is independent
and can be unpacked into the same directory. `RAW_DATA_MANIFEST.json` verifies
every asset; each ZIP's `MEMBERS.json` verifies its files and records their sources.

## Contents

- **`raw-responses-*.zip`:** lossless local response CSVs, the final collection
  and recovery records, the original four-judge API responses and request journal,
  the six-judge follow-up API upload, and source/coverage mappings. Original
  response text, failed/invalid outcomes and available finish reasons remain.
- **`prompt-contexts-*.zip`:** original canonical messages, model-templated prompt
  files, reasoning prompts, mechanism annotation units, selections, model
  configurations and attempt records from the paper's local studies. Absolute
  symlinks are recorded in `CONTEXT_SYMLINKS.json`, not recreated on your machine.
- **`trace-label-records-*.zip`:** exact written explanations supplied to the
  labelers, prompts, labels, mappings and raw batch records for saved trace-label
  revisions. The paper uses `nn-trace-labels-qwen122-revision-20261024`; other runs
  are retained as history, not pooled into that analysis.
- **Already in Git:** `data/prompts/catalog.jsonl.xz` losslessly restores the
  625,000-record API prompt catalogue. `data/paper/analysis/` provides the frozen
  per-example scores, case identities, weights and labels used in the figures.

## Read and join

Local CSVs retain the complete source record in `raw_record_json`. Columns such as
`raw_response`, `judge`, `prompt_id`, `sample_id`, `condition` and `source_path`
are convenience projections. Blank fields mean absent source metadata, not a
negative label. Preserve source paths and row numbers when joining old records.

For modern API prompts, restore the catalogue in a writable copy of `data/prompts/`
with `python restore.py`, then run `python kit.py verify`. Join API responses by
`(model, prompt_id)` to the catalogue. Original local ablation/reasoning prompt IDs
can differ from the later API catalogue: use their original prompt files under
`local/<study>/` and the recorded case/condition mapping instead of inventing IDs.
Trace annotations join by `unit_id` to `units.jsonl` and `mapping_PRIVATE.jsonl`.
Here `PRIVATE` denotes analysis metadata withheld from the judge, not an API key.

```python
import csv, gzip, json

with gzip.open('data/raw/final_local/paper-complete-20261021.raw.csv.gz', 'rt') as f:
    for row in csv.DictReader(f):
        original_record = json.loads(row['raw_record_json'])
        # Use original_record and row['source_path'] for provenance.
```

## Scope and interpretation

The final local snapshot supersedes the partial 21 October snapshot; both are
labelled. Reused responses, superseded attempts and historical CSVs can overlap.
**Row counts are not counts of independent model calls.** Apply the frozen
selection, parsing and weighting rules in `paper/` when reproducing the paper.
Additional historical CSVs are included for audit; their exploratory studies are
not added to the paper's results.

The newer API upload supplied five columns only. Its provider request journal,
exact backend versions and inference configuration were not supplied, so this
release cannot manufacture that metadata. The original four-judge journal is
included where available. Missing/blocked responses remain recorded as such.

This is a prompt/response supplement, not a whole-machine image. It excludes
model weights, activation arrays, installed environments and redundant analysis
plots. No inference was rerun to create it. Original local backups remain intact.
