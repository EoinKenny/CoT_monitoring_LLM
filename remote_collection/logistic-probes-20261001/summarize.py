"""Render the saved probe evaluation without changing any model selection."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
report=json.loads((root/'results.json').read_text())
labels={'wildguard':'WildGuard response harmfulness','high_stakes':'High-Stakes','toxicchat':'ToxicChat toxicity'}
lines=['# Mean-pooled logistic probe results','',
       'All thresholds and L2 regularization strengths were selected using validation data. Test results were computed after all three selections were frozen. Each probe uses one example-level label and mean-pooled content-token activations; padding, special tokens, and template-only positions are excluded.','',
       '| Dataset | Train | Validation | Test | Threshold | C | Accuracy | Precision | Recall | F1 | AUROC |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,r in report['results'].items():
    t=r['test'];s=r['splits']
    lines.append(f"| {labels[name]} | {s['train']['n']:,} | {s['val']['n']:,} | {s['test']['n']:,} | {r['threshold']:.6f} | {r['C']:g} | {t['accuracy']:.2%} | {t['precision']:.2%} | {t['recall']:.2%} | {t['f1']:.2%} | {t['auroc']:.4f} |")
lines+=['','## High-Stakes test breakdown','',
        'A single probe and validation-selected threshold are used across these test subsets. The combined score depends on this release’s mixture of synthetic and external examples.','',
        '| Test subset | N | Accuracy | Precision | Recall | F1 | AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|']
h=report['results']['high_stakes']
for label,t in [('Synthetic held-out',h['synthetic_test']),('External combined',h['external_test']),*h['test_by_source'].items()]:
    lines.append(f"| {label} | {t['n']:,} | {t['accuracy']:.2%} | {t['precision']:.2%} | {t['recall']:.2%} | {t['f1']:.2%} | {t['auroc']:.4f} |")
lines+=['','## Threshold comparison and class balance','',
        '| Dataset | Positive test prevalence | Majority-class accuracy | Accuracy at 0.5 | F1 at 0.5 | F1 at selected threshold |',
        '|---|---:|---:|---:|---:|---:|']
for name,r in report['results'].items():
    t=r['test'];d=r['test_at_05']
    lines.append(f"| {labels[name]} | {t['positive']/t['n']:.2%} | {r['majority_baseline_accuracy']:.2%} | {d['accuracy']:.2%} | {d['f1']:.2%} | {t['f1']:.2%} |")
lines+=['','## Split audit','',
        'Official test partitions were retained subject to explicit duplicate/conflict exclusions. WildGuard and ToxicChat validation sets come from grouped, stratified holdouts of their training data. High-Stakes uses the released external validation partitions. Exact input conflicts and duplicates are excluded; prompt/conversation/source/pair groups cannot cross retained splits. This audit does not establish absence of all semantic near-duplicates.','']
for name,r in report['results'].items():
    lines.append(f"- {labels[name]} exclusions: `{json.dumps(r['splits']['excluded'],sort_keys=True)}`.")
lines+=['','## Reproducibility','',
        f"Total run time: {report['wall_seconds']/60:.2f} minutes, including full-cache checks and content pooling.",
        '', 'See `README.md` for pooling and selection details, `results.json` for metrics and configuration, per-dataset split files for assignments, and the saved `.joblib` files for fitted scaler, logistic probe, and threshold. Held-out labels, probabilities, and predictions are stored in the test prediction archives. These are probe-only results; no LLM-judge cascade has been evaluated.','']
(root/'RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines[:10]))
