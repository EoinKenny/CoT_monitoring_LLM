import json
from pathlib import Path
R=Path(__file__).resolve().parent/'run'
d=json.loads((R/'results.json').read_text())
lines=['# Full-test Qwen cascade results','','Qwen3.6-27B-FP8 judged every positive prediction from the retained class-weighted probes: 5,190 flagged sequences, including 300 verified reused pilot decisions. All unflagged sequences retain negative decisions. Approved zero-shot prompts and validation-tuned probe thresholds were fixed throughout. vLLM 0.30.0, thinking disabled, greedy one-token choice between 0 and 1. No truncation.','','| Dataset | Probe F1 | Cascade F1 | Change (points) | Probe precision | Cascade precision | Probe recall | Cascade recall |','|---|---:|---:|---:|---:|---:|---:|---:|']
for n,r in d['results'].items():
 b,c=r['probe'],r['cascade'];lines.append(f"| {n} | {100*b['f1']:.2f}% | {100*c['f1']:.2f}% | {100*r['f1_change']:+.2f} | {100*b['precision']:.2f}% | {100*c['precision']:.2f}% | {100*b['recall']:.2f}% | {100*c['recall']:.2f}% |")
lines+=['','| Dataset | Flags judged | True positives retained | True positives rejected | False positives retained | False positives removed |','|---|---:|---:|---:|---:|---:|']
for n,r in d['results'].items():
 c=r['cascade'];lines.append(f"| {n} | {r['flagged']} | {c['tp']} | {r['true_flags_rejected']} | {c['fp']} | {r['false_flags_removed']} |")
lines+=['','These are measured full-test results, not pilot expansion estimates. They evaluate the positive-only confirmation pipeline, not a standalone judge applied to every sequence. Test labels define correctness. The same public benchmarks could have appeared in judge training; training-data independence is not established. Earlier pilot outputs were inspected, but no prompt or threshold changes were made for this full evaluation.']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
