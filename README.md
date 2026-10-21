# Nearest-neighbour explanations for probe → LLM classification

Reproduction code for the Interpretable probe study, archived on 21 October 2026.
The primary question is whether training examples retrieved in a backbone's
activation space improve a downstream LLM judge's decisions.

**This repository contains code and protocols. Raw conversations, model outputs,
case-level CSVs, annotations, fitted probes, and analysis inputs are kept in the
separate private local backup.** No inference is started by installing or testing
this repository. The active completion run was still collecting data at the
time of this snapshot; it is not a finished paper dataset.

## What is included

| Study | Source |
|---|---|
| Backbone activations, probe fitting, frozen splits and sampling | `reproduction/activation_full`, `logistic_probes`, `grid_completion_20261008`, `full_threshold_sweep_20261009` |
| Six local judges + four original API judges, three explanation conditions | `reproduction/probability_500_20261012`, `api_results_20261015` |
| Fixed probe 0.5 / judge 50 and controlled threshold tuning | `scripts/reproduce_main.py`, `reproduction/anchor_transfer_20261018` |
| Precision–recall trade-offs, including the four toxicity datasets | `reproduction/toxicity_four_20261018`, `target95_20261015` |
| NN information ablations, random reference, and 2-NN | `reproduction/nn_ablation_*`, `two_nn_20261014` |
| Probability-first / brief explanation-after and reasoning-first / probability-last | `reproduction/nn_reason_flips_20261017`, `nn_reason_first_512_20261017` |
| Labelled similarity / important contrast annotations and outcome associations | `reproduction/nn_mechanism_agreement_20261019`, `nn_mechanism_outcomes_20261019`, `nn_cosine_outcomes_20261019` |
| Logistic / k-NN selectors and exploratory follow-ups | `reproduction/nn_*gate*`, `nn_selector_*`, `nn_paper_followup_20261020` |
| Current missing-cell collection and six-API export tooling | `reproduction/paper_complete_20261021`, `remote_collection/paper-complete-20261021` |

The historic semantic-retrieval follow-up is retained for provenance; it is not
the primary activation-neighbour pipeline. Older protocols and failed/repaired
implementations are retained, not silently presented as the final method.

## Reproduce the main results without a GPU

Use Python 3.11 or newer and install `requirements-analysis.txt`. Exact recorded
environments are in `environments/`; these are provenance snapshots, whereas the
requirements file specifies the smaller analysis environment.

Restore the private backup with `scripts/restore_backup.py` (see `docs/DATA.md`).
Then run:

```bash
python scripts/reproduce_main.py \
  --data /path/to/restored/reproduction \
  --out outputs/main \
  --frontier --check-existing
```

This checks the frozen input hashes, refits thresholds on validation only, and
recomputes held-out test metrics. It writes:

- `cells.csv`: 12,000 fixed / transferred-threshold condition rows, including the
  separate common-valid sensitivity population.
- `thresholds.csv`: the actual probe and judge cutoffs for every anchor and cell.
- `summary.csv`: overall, judge, backbone and dataset macro means for all ten
  datasets and the specified four-dataset toxicity subset.
- `tradeoff_points.csv.gz`: 369,000 test operating points at thresholds selected
  by validation F-beta, when `--frontier` is supplied.
- `VERIFIED.json`: input hashes and numerical comparison with saved results.

The main command deliberately reproduces the frozen ten-judge snapshot. New API
judges and ongoing completion outputs must be integrated as a new, separately
versioned dataset, not mixed into its frozen score array.

## The comparison being reproduced

The fixed policy predicts positive only when the probe score is at least 0.5
**and** the judge's integer score is at least 50.

For tuning, each judge–backbone–dataset combination selects a probe/judge pair
on validation using one anchor condition. The same pair is then applied to
control, saliency and NN on the same test cases. Repeat with each of the three
anchors and report the within-anchor differences and their average. This keeps
referrals and cutoffs matched within each explanation comparison. The averaged
delta is a sensitivity summary, not a new deployable threshold.

The original grid includes judge cutoffs 0–101 (0 accepts all referrals; 101
rejects all). Preserve these endpoints and report their frequency. Threshold
ties use the highest probe cutoff and then highest judge cutoff within 1e-12.
The frontier display uses validation-selected operating points evaluated on
test; it must not be described as a test-selected optimum.

Design weights, paired cases, and API invalid/missing outcomes remain explicit.
The primary descriptive means combine local full-test estimates and API
matched-valid estimates; the all-judge-common-valid sensitivity is separate.
The four-dataset subset is Aegis, BeaverTails, ToxicChat and ToxiGen.

## Other studies and collection

`reproduction/` preserves local orchestration and analysis sources.
`remote_collection/` preserves the exact server collection code and protocols.
Their hashes are indexed in `docs/SOURCE_MANIFEST.json` and
`docs/REMOTE_SOURCE_MANIFEST.json`. Historical scripts retain their original
paths for source fidelity; see `docs/REPRODUCING_LEGACY.md` before running them.
Do not blindly launch old controllers: they may point at an already attempted
experiment. No numerical replay-agreement gates should be reintroduced.

Main probability classification is greedy with native thinking disabled. The
reasoning studies ask for **expressed explanations** in different output orders;
they are not measurements of a model's private internal reasoning. See the
frozen protocols for caps, parsing and selection. Invalid, capped, interrupted
and ambiguous attempts are observations to retain, not invitations to retry.

## Interpretation and release limits

These datasets and test sets have already been explored. Report exploratory
follow-ups honestly. LLM mechanism annotations and expressed explanations
provide associations, not proof of causal internal mechanisms. Assistant-written
case audits are not independent human annotation. Model aliases do not establish
provider versions: retain exact backend manifests and request journals.

This is a private research archive, not a data redistribution licence. Original
dataset, model and upstream-code licence obligations remain applicable. Large
model weights and activation caches are not in Git or the compact backup;
regenerating those requires the recorded upstream models, datasets and GPU setup.
