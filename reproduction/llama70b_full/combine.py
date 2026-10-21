from common import *
import csv,numpy as np
old=read(PRIOR/'RESULTS.json');new=read(ROOT/'RESULTS.json');merged={k:old[k]+new[k] for k in ['judge_results','probe_results','paired_comparisons']};assert len(merged['judge_results'])==600 and len(merged['probe_results'])==120
merged['sources']={str(p):sha(p) for p in [PRIOR/'RESULTS.json',ROOT/'RESULTS.json']};write(ROOT/'COMBINED_RESULTS.json',merged)
for key in ['judge_results','probe_results']:
 rows=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in merged[key]]
 with (ROOT/('combined_'+key+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Four-backbone comparison','', 'Exploratory, design-weighted full-test pipeline estimates where flags were sampled. Each backbone has its own frozen flagged sample. Differences across backbones are not matched-case causal effects of model size.','', '| Backbone | Judge | Baseline F1 | Saliency F1 | NN F1 |','|---|---|---:|---:|---:|']
for b in ['llama','qwen','gemma','llama70']:
 for m in JUDGES:
  means=[np.mean([r['pipeline_f1'] for r in merged['judge_results'] if r['backbone']==b and r['judge']==m and r['condition']==c]) for c in CONDITIONS];lines.append('| '+b+' | '+m+' | '+' | '.join(f'{x:.4f}' for x in means)+' |')
(ROOT/'COMBINED_RESULTS.md').write_text('\n'.join(lines)+'\n');write(ROOT/'ALL_COMPLETED.json',dict(judge_cells=600,backbones=4,probe_summaries=40,sha256=sha(ROOT/'COMBINED_RESULTS.json'),time=time.time()))
