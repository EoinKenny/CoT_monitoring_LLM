from common import *
import numpy as np,html,csv

def main():
 summaries=[];pairs=[];effects=[];sources={};timings=[]
 for b in BBS:
  for j in ALL:
   d=ROOT/'analysis'/f'{b}_{j}';done=read(d/'COMPLETED.json')
   for name,h in done['files'].items():assert sha(d/name)==h
   sources[b+'_'+j]=sha(d/'COMPLETED.json')
   for r in read(d/'summary.json'):summaries.append(dict(backbone=b,judge=j,**r))
   for r in read(d/'pairs.json'):pairs.append(dict(backbone=b,judge=j,**r))
   for r in read(d/'comparisons.json'):effects.append(dict(backbone=b,judge=j,**r))
   for r in read(d/'runtime.json'):timings.append(dict(backbone=b,judge=j,**r))
 out=ROOT/'results';out.mkdir(exist_ok=True);macro=[]
 for dimension,groups in [('overall',['all']),('judge',ALL),('backbone',BBS)]:
  for group in groups:
   rr=[r for r in pairs if dimension=='overall' or r[dimension]==group]
   for policy in ['fixed50','joint']:
    for c in ['baseline','nearest','nearest_reason_after']:
     subset=[r for r in rr if r['condition']==c and r['policy']==policy]
     m=dict(dimension=dimension,group=group,condition=c,policy=policy,cells=len(subset))
     for k in ['f1','precision','recall','routing_rate','input_tokens_per_population_example']:m[k]=float(np.mean([r[k] for r in subset]))
     macro.append(m)
 for name,x in [('summary',summaries),('pairs',pairs),('comparisons',effects),('runtime',timings),('macro',macro)]:write(out/(name+'.json'),x)
 with (out/'macro.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(macro[0]));w.writeheader();w.writerows(macro)
 table='<table><tr>'+''.join('<th>'+h+'</th>' for h in ['Group','Condition','Policy','F1','Precision','Recall','Input tokens/example'])+'</tr>'
 for r in macro:table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['dimension']+': '+r['group'],r['condition'],r['policy'],*[f'{100*r[k]:.2f}' for k in ['f1','precision','recall']],f'{r["input_tokens_per_population_example"]:.1f}'])+'</tr>'
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Brief explanation comparison</title><style>body{font:16px system-ui;margin:40px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}</style><h1>Probability first, brief explanation afterward</h1><p>Five backbones, six judges, ten datasets. 500 validation and 500 test cases per dataset/backbone; shared across judges and conditions. Equal-cell macro averages of weighted test estimates. Both tuning policies choose the probe gate using validation only; joint also chooses the judge cutoff. Baseline and one-NN responses are reused with verified provenance. Per-pair reports contain paired stratified intervals conditional on fitted thresholds; no pooled significance claim is made from independent-cell errors. Neighbour-alignment and crossed-threshold diagnostics distinguish score changes from policy changes.</p>'+table+'</table>')
 write(ROOT/'COMPLETED.json',dict(time=time.time(),new_judgments=300000,reused_judgments=600000,sources=sources,files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':main()
