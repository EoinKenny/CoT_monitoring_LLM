import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'qwen36_v1'
assert (RUN/'COMPLETED').exists()
private=json.loads((ROOT/'sample_private.json').read_text())
judgments=[json.loads(x) for x in (RUN/'judgments.jsonl').read_text().splitlines()]
byid={r['sample_id']:r for r in judgments}
assert len(byid)==len(judgments)==300
baseline=json.loads((ROOT/'weighted_results.json').read_text())['results']
def metrics(tp,fp,fn,tn):
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),accuracy=(tp+tn)/(tp+fp+fn+tn))
results={}
for name in ['wildguard','high_stakes','toxicchat']:
 rows=[r for r in private['records'] if r['dataset']==name]
 for r in rows:assert byid[r['sample_id']]['dataset']==name
 a=sum(byid[r['sample_id']]['prediction'] for r in rows if r['true_label']==1)
 b=sum(byid[r['sample_id']]['prediction'] for r in rows if r['true_label']==0)
 base=baseline[name]['test']
 tp=base['tp']*a/50;fp=base['fp']*b/50
 fn=base['fn']+base['tp']-tp;tn=base['tn']+base['fp']-fp
 full=metrics(tp,fp,fn,tn)
 results[name]=dict(retained_true_flags=a,rejected_true_flags=50-a,retained_false_flags=b,rejected_false_flags=50-b,sample_probe=metrics(50,50,0,0),sample_cascade=metrics(a,b,50-a,50-b),full_test_probe=base,estimated_full_test_cascade=full,estimated_full_test_f1_change=full['f1']-base['f1'])
report=dict(results=results,n=300,notes=['Sample F1 covers only the balanced flagged subset, not the whole test set.','Full-test cascade confusion counts are stratified expansion estimates; FN/TN among unflagged examples are retained exactly.','No judge outputs were used to tune prompts, probe settings, or thresholds.','Finite-sample estimates; no claim of statistically significant improvement.','Evaluation uses hard binary next-token choice, no rationale generation or probability threshold tuning.'])
(RUN/'results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
