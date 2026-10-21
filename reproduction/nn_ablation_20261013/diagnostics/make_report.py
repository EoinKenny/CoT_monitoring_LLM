import json,html
from pathlib import Path
D=Path(__file__).resolve().parent;data=json.loads((D/'case_analysis.json').read_text());rows=json.loads((D/'cases.json').read_text())['rows']
esc=html.escape
examples=[]
for prefix in ['toxicchat:535d027','toxicchat:c1bf8b','toxicchat:c8d890']:
 r=next(r for r in rows if r['sample_id'].startswith(prefix))
 examples.append('<article><h3>'+esc(r['sample_id'][:23])+'</h3><p><b>Target (dataset class '+str(r['true_label'])+'):</b> '+esc(r['messages'][0]['content'])+'</p><p><b>Neighbour (label '+str(r['nearest_label'])+', prediction '+str(r['nearest_prediction'])+'):</b> '+esc(r['nearest_messages'][0]['content'])+'</p><p>Probe score '+f"{r['probe_probability']:.5f}"+'; full-NN probability '+str(r['scores']['nearest'])+'; text + prediction probability '+str(r['scores']['text_prediction'])+'.</p></article>')
tables=[]
for mode in ['joint_pipeline','judge_at50','same_baseline_gate_at50']:
 t='<h2>'+esc(mode)+'</h2><table><tr><th>Scope</th><th>Alignment</th><th>Sample n</th><th>Corrected</th><th>Introduced</th><th>Weighted net accuracy change (pp)</th></tr>'
 for r in data['random_alignment']:
  if r['mode']==mode:t+='<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [r['scope'],r['alignment'],r['n'],r['helped'],r['hurt'],round(r['weighted_net_accuracy_pp'],3)])+'</tr>'
 tables.append(t+'</table>')
report='''<!doctype html><meta charset="utf-8"><title>ToxicChat and random-reference diagnostics</title><style>body{font:17px system-ui;line-height:1.5;max-width:1100px;margin:40px auto;color:#18202c}table{border-collapse:collapse;margin:20px 0}td,th{padding:9px;border:1px solid #ddd;text-align:right}article{padding:15px;border:1px solid #ccd;margin:16px 0}th{background:#eef2f7}</style>
<h1>ToxicChat and random-reference diagnostics</h1>
<p>Llama 3.3 70B backbone and Qwen 3.5 122B judge. Exact paired frozen test cases: 500 ToxicChat; 5,000 across ten datasets. No new model calls. Comparisons use dataset gold labels, including potentially debatable annotations.</p>
<h2>Main finding: the ToxicChat gain is mostly the selected operating point</h2>
<p>Full NN and text + prediction return identical scores on 465/500 ToxicChat test cases. The neighbour label and probe prediction agree in all 500 cases. With the separately validation-selected settings, text + prediction corrects 24 cases and introduces 6 errors relative to full NN. Of the 24 corrections, 19 are newly referred cases; 18 have identical judge scores across prompts.</p>
<table><tr><th>Judge scores</th><th>Probe gate</th><th>Judge cutoff</th><th>Test F1</th></tr>
<tr><td>Full NN</td><td>0.75190</td><td>15</td><td>79.96%</td></tr>
<tr><td>Text + prediction</td><td>0.75190</td><td>15</td><td>80.53%</td></tr>
<tr><td>Full NN</td><td>0.01325</td><td>95</td><td>87.20%</td></tr>
<tr><td>Text + prediction</td><td>0.01325</td><td>95</td><td>85.79%</td></tr></table>
<p>The diagonal entries are the original validation-selected policies. The crossed settings are post-hoc diagnostic comparisons, not newly selected configurations. Full-NN validation F1 is 84.97% at its own policy and 83.70% at the other policy; text + prediction validation F1 is 83.73% at the full-NN policy and 84.82% at its own policy. Modest changes in validation scores led to very different routing policies.</p>
<p>At a shared judge cutoff of 50, before the probe gate, removing the label corrects 3 test cases but introduces 6 errors. Its apparent 5.83-point pipeline F1 advantage does not establish better stand-alone ToxicChat judgments.</p>
<h2>Examples</h2>'''+''.join(examples)+'''
<p>The first example gets 100 from both prompts but is referred only by the lower probe gate. The second gets 15 from both and changes class because the judge cutoff changes. The third changes score from 15 to 0, a genuine local reduction in false-positive scoring; the reference label and prediction are both 1, so this is not a disagreement-resolution case.</p>
<h2>Random-reference alignment</h2>
<p>Aligned means the randomly selected training example's dataset label equals the target gold label. Corrected/introduced are raw matched case counts relative to no explanation. Weighted net accuracy changes use inverse inclusion probabilities and pool across dataset populations; these are not the equal-dataset macro F1 used in the main report. joint_pipeline uses each condition's validation-selected policy. judge_at50 scores every sampled case at the same cutoff without a probe gate. same_baseline_gate_at50 holds both the gate and judge cutoff equal across conditions.</p>'''+''.join(tables)+'''
<p>ToxicChat has no random-reference label/prediction disagreements. Across ten datasets there are 89/5,000; defining alignment using the random reference's probe prediction instead gives a similar pooled weighted pattern (judge-only cutoff50 net accuracy: +1.52 pp aligned, +1.55 pp diverged). The aligned-reference pattern in ToxicChat does not generalize uniformly across all tasks.</p>
<p>Interpretation: routing and decision-threshold selection explain most of the large ToxicChat contrast. Some individual score changes are consistent with reference influence, but these outputs do not reveal the model's internal reasoning. Random examples can help without matching the target class, suggesting task illustration or contrast can contribute alongside label influence. These are exploratory diagnostics on one model pair, with one frozen random-reference draw per case.</p>'''
(D/'report.html').write_text(report)
