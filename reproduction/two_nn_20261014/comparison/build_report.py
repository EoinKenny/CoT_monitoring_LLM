from pathlib import Path
import csv,json,hashlib,html
R=Path(__file__).resolve().parents[1];P=R.parent/'probability_500_20261012/results/judge_summary.csv';O=Path(__file__).resolve().parent
rows=list(csv.DictReader((R/'results/macro.csv').open())); prior=list(csv.DictReader(P.open()))
lookup={(x['judge'],x['variant']):x for x in prior}; js=['qwen','gemma','mistral','olmo','glm','qwen122']; metrics=['f1','precision','recall','routing_rate']; out=[]
for policy in ['fixed50','joint']:
 for j in js:
  for c in ['baseline','saliency','nearest','nearest2']:
   if c=='saliency':
    x=lookup[j,c+'_'+policy];v={k:float(x[k]) for k in metrics}
   else:
    x=next(x for x in rows if x['dimension']=='judge' and x['group']==j and x['condition']==c and x['policy']==policy);v={k:float(x[k]) for k in metrics}
    if c!='nearest2':
     for k in metrics:assert abs(v[k]-float(lookup[j,c+'_'+policy][k]))<1e-12
   out.append(dict(judge=j,policy=policy,condition=c,**v))
 for c in ['baseline','saliency','nearest','nearest2']:
  rr=[x for x in out if x['policy']==policy and x['condition']==c and x['judge']!='average']
  out.append(dict(judge='average',policy=policy,condition=c,**{k:sum(x[k] for x in rr)/6 for k in metrics}))
with (O/'four_conditions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
names=dict(qwen='Qwen27B',gemma='Gemma31B',mistral='Mistral',olmo='OLMo',glm='GLM',qwen122='Qwen122B',average='Average')
h=['<!doctype html><meta charset="utf-8"><title>Four explanation conditions</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#182332}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f8}p{line-height:1.6}</style><h1>Control, saliency, 1 NN and 2 NN</h1><p>Six local judges, five backbones and ten datasets. Each judge uses the same 500 validation and 500 test cases per backbone–dataset pair. Values are equal-cell macro averages of weighted held-out test estimates. No new model calls were made for this report.</p><p>Thresholds are chosen on validation only. Fixed-50 keeps the judge threshold at 50 while tuning the probe gate; joint tuning chooses both. The tiny overall difference between one and two neighbours is descriptive, not a pooled significance claim.</p>']
for policy in ['joint','fixed50']:
 h.append('<h2>'+('Both thresholds tuned' if policy=='joint' else 'Judge threshold fixed at 50')+'</h2>')
 for m in ['f1','precision','recall']:
  h.append('<h3>'+m.upper()+'</h3><table><tr><th>Judge</th><th>Control</th><th>Saliency</th><th>1 NN</th><th>2 NN</th><th>2 NN − 1 NN (points)</th></tr>')
  for j in js+['average']:
   v={x['condition']:x[m]*100 for x in out if x['judge']==j and x['policy']==policy};h.append('<tr><td>'+names[j]+'</td>'+''.join('<td>'+f'{v[c]:.2f}'+'</td>' for c in ['baseline','saliency','nearest','nearest2'])+'<td>'+f'{v["nearest2"]-v["nearest"]:+.2f}'+'</td></tr>')
  h.append('</table>')
h.append('<p><a href="../results/report.html">Original zero/one/two-neighbour report, including input-token cost</a> · <a href="four_conditions.csv">Download all metrics</a></p>')
(O/'report.html').write_text(''.join(h))
receipt={'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,R/'results/macro.csv']},'baseline_and_nearest_match_parent':True,'rows':len(out),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'report.html',O/'four_conditions.csv']}}
(O/'VERIFIED.json').write_text(json.dumps(receipt,indent=2));print('Four-condition report created; reused control/1-NN metrics match exactly.')
