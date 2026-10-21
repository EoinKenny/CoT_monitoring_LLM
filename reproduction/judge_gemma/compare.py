from pathlib import Path
import json
R=Path(__file__).resolve().parent
G=json.loads((R/'run/results.json').read_text())['results']
Q=json.loads((R.parent/'judge_full/run/results.json').read_text())['results']
rows=['# Full-test judge comparison','','All 5,190 flagged sequences judged by each model. Same fixed probe thresholds, rubric text, conversations, and greedy one-token binary decision rule. Both judges have thinking disabled. Native model chat templates and tokenizers differ. Qwen3.6-27B-FP8 compared with Gemma 4 31B Instruct official QAT W4A16 checkpoint, using vLLM 0.30.0. Quantization differs, so this compares the deployed configurations rather than isolating model family.','','| Dataset | Probe F1 | Qwen cascade F1 | Gemma cascade F1 | Gemma minus probe | Gemma minus Qwen |','|---|---:|---:|---:|---:|---:|']
for name,g in G.items():
 q=Q[name];assert g['probe']==q['probe'];b=g['probe']['f1'];gf=g['cascade']['f1'];qf=q['cascade']['f1']
 rows.append(f'| {name} | {b*100:.2f}% | {qf*100:.2f}% | {gf*100:.2f}% | {(gf-b)*100:+.2f} points | {(gf-qf)*100:+.2f} points |')
rows+=['','| Dataset | Gemma precision | Gemma recall | False flags removed | Genuine positives rejected |','|---|---:|---:|---:|---:|']
for name,g in G.items():
 c=g['cascade'];rows.append(f"| {name} | {c['precision']*100:.2f}% | {c['recall']*100:.2f}% | {g['false_flags_removed']} / {g['probe']['fp']} | {g['true_flags_rejected']} / {g['probe']['tp']} |")
rows+=['','These are measured complete test-set cascade results. Unflagged negatives bypass both judges. Public-benchmark exposure during judge training is unknown. Prompts were not tuned on Gemma outputs. Detailed conversation records remain on the remote instance; this local report contains aggregate metrics only.']
(R/'COMPARISON.md').write_text('\n'.join(rows)+'\n')
print('\n'.join(rows))
