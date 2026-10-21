import json
from pathlib import Path
import sys
R=Path(sys.argv[1]);run=R/'run'
assert (run/'COMPLETED').exists()
rows=json.loads((R/'private.json').read_text());js=[json.loads(s) for s in (run/'judgments.jsonl').read_text().splitlines()];byid={r['sample_id']:r for r in js}
assert len(js)==len(byid)==len(rows) and set(byid)=={r['sample_id'] for r in rows}
bases=json.loads((R/'weighted_results.json').read_text())['results']
def metrics(tp,fp,fn,tn):
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,n=tp+fp+fn+tn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),accuracy=(tp+tn)/(tp+fp+fn+tn),balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),macro_f1=.5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn)))
results={}
for name,base in bases.items():
 subset=[r for r in rows if r['dataset']==name];b=base['test'];tp=fp=0
 for r in subset:
  pred=byid[r['sample_id']];assert pred['dataset']==name and pred['prediction'] in [0,1]
  tp+=int(r['true_label']==1 and pred['prediction']==1);fp+=int(r['true_label']==0 and pred['prediction']==1)
 assert len(subset)==b['tp']+b['fp']
 c=metrics(tp,fp,b['fn']+b['tp']-tp,b['tn']+b['fp']-fp)
 results[name]=dict(probe=b,cascade=c,flagged=len(subset),true_flags_rejected=b['tp']-tp,false_flags_removed=b['fp']-fp,f1_change=c['f1']-b['f1'])
(run/'results.json').write_text(json.dumps(dict(results=results,judgments=len(rows),reused=0,scope='Measured full-test cascade; all flags judged; unflagged negatives unchanged; frozen approved prompts and probe thresholds'),indent=2))
print(json.dumps(results,indent=2))

