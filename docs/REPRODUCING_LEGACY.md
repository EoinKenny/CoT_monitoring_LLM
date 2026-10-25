> Current release: use `python reproduce_results.py`. This document describes
> the historical ten-judge archive; see [current reproduction](REPRODUCTION.md).

# Historical experiment sources

The portable main evaluator is `scripts/reproduce_main.py`. The historic sources
are preserved separately so that prompts, parsers, model settings and failed
attempts can be audited without silently rewriting the experiment.

Most local analysis scripts use a `reproduction/` directory with sibling study
folders. Some verification receipts contain original absolute paths. To run
those sources without changing their provenance, use an isolated container or
machine whose restored data is mounted at the historical aliases:

```
/Users/eoinkenny/.codex/.chatgpt-projects/g-p-6ab51e7550bc81919a7e0121a4671c91/reproduction
/home/ubuntu/experiments
```

The first maps to the restored `reproduction` directory; the second maps to
`remote_experiments`. Do this in a scratch copy, because older analysis programs
write reports in place. Do not mount the only verified archive writable. Some
GPU sources also reference installed virtual environments and downloaded model
locations; use the environment records and per-run configs to reconstruct those
dependencies. An environment package inventory does not include model weights.

Core historical entry points, in dependency order:

1. `api_results_20261015/integrate.py`: join the original API responses to frozen
   design and local scores, preserving parse/missingness rules.
2. `two_policy_evaluation_20261018/analyze.py`: fixed and independently selected
   policies, validation-selected F-beta points, retained as the source audit.
3. `anchor_transfer_20261018/analyze.py`: shared threshold-pair comparisons for
   each anchor. This is the controlled explanation comparison used in the paper.
4. `toxicity_four_20261018/analyze.py`, `plot_tradeoffs.py`, and
   `fixed_probe_tradeoffs.py`: the requested subset and precision–recall figures.
5. `nn_mechanism_agreement_20261019/analyze.py` and
   `nn_mechanism_outcomes_20261019`: label agreement, helpful/harmful/unchanged
   outcomes and associations.
6. `nn_reason_flips_20261017/final_analysis` and
   `nn_rationale_mechanisms_20261020`: explanation-order and source-grounded
   expressed-rationale analyses. Per-case coding inputs live in the private
   backup, including data-bearing Python annotation records.

The independent-analysis histories, older sample-size experiments, semantic
retrieval comparisons and selector searches remain exploratory. Their presence
in this archive does not make them part of a preregistered primary analysis.

For collection, inspect `remote_collection/<experiment>/PROTOCOL.json`, its
launch/config records in the data backup, and its controller before use. Preserve
native-thinking settings, token caps, sampling seeds, exact prompt messages,
attempt journals and response hashes. Never restart a historic controller over
existing outputs or silently retry an ambiguous request. New model calls require
an explicitly new experiment root and an intentional protocol decision.
