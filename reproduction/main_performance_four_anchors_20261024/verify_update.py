import csv,gzip,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=R.parent/'main_performance_1nn_2nn_all12_20261023/results'
NEW=R/'results'
def rows(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
  yield from csv.DictReader(f)
counts={}
roundoff={}
numeric_metrics={'tp','fp','fn','tn','precision','recall','f1','accuracy','referral_rate','validation_f1'}
for name in ['cells.csv','thresholds.csv','tradeoff_points.csv.gz','fixed_probe_curves.csv.gz','coverage.csv','probe_only.csv']:
 old=iter(rows(OLD/name));n=0;added=0
 for row in rows(NEW/name):
  if row.get('anchor')=='nearest':added+=1;continue
  prior=next(old)
  for field,value in row.items():
   if value==prior[field]:continue
   assert field in numeric_metrics,(name,n,field,value,prior[field])
   delta=abs(float(value)-float(prior[field]))
   roundoff[field]=max(roundoff.get(field,0),delta)
   assert delta<1e-9,(name,n,field,delta)
  n+=1
 assert next(old,None) is None,name
 counts[name]={'unchanged_rows':n,'new_1nn_anchor_rows':added}
assert counts['cells.csv']=={'unchanged_rows':9600,'new_1nn_anchor_rows':2400}
assert counts['thresholds.csv']['new_1nn_anchor_rows']==600
assert counts['tradeoff_points.csv.gz']['new_1nn_anchor_rows']==98400
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
receipt={'checks':counts,'max_absolute_floating_point_differences':roundoff,'source_scores_sha256':sha(R.parent/'main_performance_1nn_2nn_all12_20261023/scores.npz'),'anchors':['baseline','saliency','nearest','nearest2'],'fixed50_and_previous_anchor_thresholds_membership_counts_exactly_unchanged':True,'aggregate_metrics_reproduced_with_floating_point_roundoff':True,'new_inference_calls':0}
(NEW/'FOUR_ANCHOR_VERIFICATION.json').write_text(json.dumps(receipt,indent=2))
config={'updated':'2026-10-24','user_instruction':'Use all four conditions as validation-tuning anchors from now on.','default_performance_analysis':str(R),'summary_policy':'average_of_four','anchors':['baseline','saliency','nearest','nearest2'],'method':'For each judge/backbone/dataset and anchor, choose probe and judge cutoffs on validation only; apply the same pair to all four explanation conditions on test; average all four anchor results and paired deltas equally.','fixed_policy':'probe 50 / judge 50','genre_groups':{'toxicity_harmful_content':['aegis','beavertails','toxicchat','toxigen'],'safety_jailbreaks':['wildguard','wildjailbreak'],'hallucination':['fava','halueval','ragtruth'],'high_stakes':['high_stakes']},'historical_results':'Earlier three-anchor reports and paper draft are preserved historical versions; their tuned numbers need updating before reuse.'}
(R.parent/'CURRENT_PERFORMANCE_ANALYSIS.json').write_text(json.dumps(config,indent=2))
print(json.dumps(receipt,indent=2))
