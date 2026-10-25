# Reproduction release — 25 October 2026

`reproduce_results.py` is the current entry point. The older
`scripts/reproduce_main.py` reproduces the archived ten-judge/three-condition
snapshot; it is not the paper's current analysis.

## Inputs and outputs

| Reference PDF page | Source data under `data/paper/` | Regenerated figure |
|---|---|---|
| 1: four-condition performance | `plots/main_cells.csv.gz`, `main_curves.csv.gz`; underlying `analysis/main_scores.npz`, `design.json.gz` | `overall_four_panel.pdf` |
| 2: nine-condition ablations | `plots/ablation_cells.csv.gz`; `analysis/ablation_scores.npz` | `overall_metrics.pdf` |
| 3 left: pair-label associations | `plots/label_cells.csv.gz`, `label_prevalence.json`; `analysis/paired_test.csv.gz` | `label_prevalence_judge_dataset.pdf` |
| 3 right: exploratory contrast | `plots/contrast_bins.json`; `analysis/paired_test.csv.gz` | `overall_contrast_net_square.pdf` |
| 4: manifold | deterministic schematic, seed 42; not measured activations | `manifold_background.pdf` |
| 5 | appendix divider | `appendix_divider.pdf` |
| 6: ablation by genre | same ablation data | `genre_f1_differences.pdf` |
| 7: expressed comparison agreement | `plots/trace_associations.json`, `trace_joints.json`; `analysis/trace_pairs.csv.gz` | `agreement_matrices.pdf` |

The combined `figures.pdf` contains all seven pages. The diagrams are re-rendered
from numeric data, not extracted screenshots. PDF page placement approximates the
hand-arranged reference; numerical values, colours, axis limits, and panel contents
are preserved. Font rendering can differ between plotting-library versions.
The eight extra appendix grids are in `all_eight_comparisons.pdf`.

`data/paper/MANIFEST.json` records SHA-256 hashes and sizes. The default command
verifies every frozen data file before plotting. `--recompute` rebuilds threshold
selection, per-cell metrics, label associations and bootstrap intervals from
per-case inputs. It writes a separate output tree; it never edits bundled data.

## Method preserved

- Main score columns: **baseline, saliency, nearest2, nearest**. Do not infer
  column order from the legend. Arrays are 50,000 cases × four conditions per judge,
  in `design.json.gz` backbone/sample order. Ablation columns are documented in
  `paper/recompute.py` and include the same baseline and full 1-NN.
- Samples: 500 validation and 500 test cases per backbone–dataset pair. Ten
  datasets × five backbones; repeated targets are not independent observations.
- Missing/invalid probabilities are `-1`. Match all four (or nine ablation)
  conditions **within a judge**, never discard a case from other judges because
  one model failed. Local judgments have complete coverage. Local metrics retain
  the original population denominators; API metrics describe the weighted
  matched-valid cohort. Both retain inverse-inclusion weights.
- Fixed performance uses probe ≥0.5 AND judge ≥50. The fixed-probe *curve* sweeps
  the judge cutoff 0–101, with 50 marked. Precision for no positive predictions
  follows the recorded zero convention.
- Tuned comparisons select a probe/judge pair on validation, transfer that same
  pair to all four explanation conditions, and average all **four** anchors:
  control, saliency, 1-NN, 2-NN. F-beta curves use 41 objectives, beta 0.1–10; beta=1
  is marked. No thresholds are selected on test. Cutoff-zero endpoints are
  retained, not silently discarded. Ties follow the frozen numerics code.
- Swarm dots average five backbones within each judge–dataset combination.
  Tuned dots also average four anchors. Means include points outside the zoom.
- Mechanism outcomes compare direct control and 1-NN judge decisions at 50,
  **without a probe-referral gate**. Original pair labels use strict majority
  consensus excluding the outcome judge; codes 0=no, 1=yes, 2=uncertain,
  3=unresolved/missing. Both labels may be yes. Exclude uncertain labels separately
  for each feature. Per-cell prevalence is design-weighted, then averaged over
  available backbones, datasets within judge, and judges. Dots are judge–dataset
  means, not individual targets.
- Label and contrast error bars use 1,000 target-ID bootstrap resamples within
  dataset (seed 20261024), shared across judges/backbones. The contrast plot
  conditions on contrast=yes and original judge probability 50–70, chosen after
  exploration; bins are [0,10), …, [90,100]. Its net gain is corrections minus
  harms, including unchanged cases in denominators. This is exploratory.
- Trace matrices use Qwen122 labels of **probability-first, explanation-after**
  written explanations on originally flipped cases. The source judge is excluded
  from original consensus. Resolved, assessable pairs only: 44,083 similarity and
  43,441 contrast pairs. Weighted joint proportions are averaged hierarchically
  before conditioning each row. The displayed raw n is not a weighted percentage.
  Expressed explanations are imperfect evidence, not private reasoning or proof
  of a causal mechanism.

## Models and dataset genres

Aliases preserve submitted model names; API aliases do not establish exact
provider versions. Request/configuration records remain authoritative.
Local judges: Qwen27B, Qwen122B, Gemma31B, Mistral119B, OLMo32B, GLM Flash.
API aliases: Sonnet4.5, Opus4.8, Opus5.5, GPT-4o, GPT Astra, GPT-5.5.
Backbones: Llama1B, Qwen4B, Gemma31B, Llama70B, Qwen122B.

- Toxicity / harmful content: Aegis, BeaverTails, ToxicChat, ToxiGen.
- Safety / jailbreaks: WildGuard, WildJailbreak.
- Hallucination: FAVA, HaluEval, RAGTruth.
- High-stakes: High-stakes.

## Scope of the archive

`reproduction/` and `remote_collection/` preserve experiment source history,
including ablations, 2-NN, both response orders, mechanism labelling, selectors,
and exploratory analyses. They are not all paper figures and many historical
scripts retain machine-specific paths. `docs/release/SOURCE_MANIFEST.json` indexes
this snapshot. The portable `paper/` package is the supported plotting path.

No model weights or activation caches are necessary for the plots. Model-query
mode reuses frozen targets, reference text, and probe outputs. Regenerating
activations/training probes requires the upstream datasets/models and GPU setup
in the archived extraction and probe scripts; it is not a one-command CPU step.
The private repository does not grant redistribution rights over upstream datasets
or models. Keep their licences and access conditions when releasing publicly.
