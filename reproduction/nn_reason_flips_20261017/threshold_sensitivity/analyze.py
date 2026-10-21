"""Offline frozen-cutoff sensitivity; no threshold fitting or inference."""
import json,gzip,hashlib,collections,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
F=ROOT/'api_results_20261015/frozen'
def read(p):
 return json.load(gzip.open(p,'rt') if str(p).endswith('.gz') else open(p))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
manifest=read(F/'EXPORT_MANIFEST.json'); checks={}
for name in ['design.json.gz','local_scores.jsonl.gz']:
 checks[name]=sha(F/name);assert checks[name]==manifest['files'][name]
design=read(F/'design.json.gz')
cases={(b,r['sample_id']):r for b,rows in design['samples'].items() for r in rows if r['split']=='test'}
pairpath=ROOT/'probability_500_20261012/results/pairs.json'
pairs=read(pairpath);thresholds={(r['judge'],r['backbone'],r['dataset']):r['variants'] for r in pairs}
snapshotpath=ROOT/'nn_reason_flips_20261017/FINAL_ANALYSIS_SNAPSHOT.json.gz'
reasons={(r['judge'],r['backbone'],r['sample_id']):r for r in read(snapshotpath)['rows']}
scores={}
with gzip.open(F/'local_scores.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line)
  if r['condition'] not in ['baseline','nearest'] or (r['backbone'],r['sample_id']) not in cases:continue
  key=(r['model'],r['backbone'],r['sample_id']);d=scores.setdefault(key,{})
  assert r['condition'] not in d
  d[r['condition']]=r['positive_probability']
assert len(cases)==25000 and len(scores)==150000 and len(reasons)==35919
stats=collections.defaultdict(collections.Counter)
def add_effect(d,old,new,y,w):
 d['n']+=1;d['weight']+=w
 if old!=new:
  event='corrected' if new==bool(y) else 'undone'
  d[event]+=1;d[event+'_weight']+=w
for key,s in scores.items():
 j,b,sid=key;r=cases[b,sid];y=bool(r['true_label']);w=1/r['inclusion_probability'];v=thresholds[j,b,r['dataset']]
 assert len(s)==2
 rr=reasons.get(key);has=rr is not None and rr['reason_probability'] is not None
 if rr:
  assert rr['control_probability']==s['baseline'] and rr['nn_probability']==s['nearest']
  for k in ['true_label','probe_probability','inclusion_probability','dataset']:assert rr[k]==r[k],(key,k)
 fixedflip=(s['baseline']>=50)!=(s['nearest']>=50)
 assert fixedflip==(rr is not None),key
 for policy in ['fixed50_judge','tuned_judge','tuned_pipeline']:
  def dec(c,score):
   t=50 if policy=='fixed50_judge' else v[c+'_joint']['probability_threshold']
   return score>=t and (policy!='tuned_pipeline' or r['probe_probability']>=v[c+'_joint']['probe_threshold'])
  control=dec('baseline',s['baseline']);nn=dec('nearest',s['nearest']);flip=control!=nn
  for group in [j,'ALL']:
   d=stats[group,policy,'original'];d['n']+=1
   if flip:
    d['flips']+=1;d['helpful' if nn==y else 'harmful']+=1;d['covered']+=has;d['overlap_fixed50']+=fixedflip
   if has:
    new=dec('nearest',rr['reason_probability'])
    add_effect(stats[group,policy,'same_collected_cohort'],nn,new,y,w)
    if flip:add_effect(stats[group,policy,'covered_policy_flips'],nn,new,y,w)
# Additional shared-cutoff sensitivity on the same collected cohort.
for rr in reasons.values():
 if rr['reason_probability'] is None:continue
 for cut in [10,25,50,75,90]:
  for group in [rr['judge'],'ALL']:
   add_effect(stats[group,'common_'+str(cut),'same_collected_cohort'],rr['nn_probability']>=cut,rr['reason_probability']>=cut,bool(rr['true_label']),1/rr['inclusion_probability'])
rows=[]
for (j,p,c),d in sorted(stats.items()):
 z=dict(judge=j,policy=p,cohort=c,**d)
 if c!='original':
  z['net']=d['corrected']-d['undone'];z['delta_accuracy_pp']=100*z['net']/d['n'] if d['n'] else None
  z['weighted_delta_accuracy_pp']=100*(d['corrected_weight']-d['undone_weight'])/d['weight'] if d['weight'] else None
 else:z['coverage_pct']=100*d['covered']/d['flips'] if d['flips'] else None
 rows.append(z)
context=[]
for j in sorted({p['judge'] for p in pairs}):
 pp=[p for p in pairs if p['judge']==j];z={'judge':j}
 for policy in ['fixed50','joint']:
  for c in ['baseline','nearest']:z[c+'_'+policy+'_f1']=sum(p['variants'][c+'_'+policy]['f1'] for p in pp)/len(pp)
  z[policy+'_nn_gain_pp']=100*(z['nearest_'+policy+'_f1']-z['baseline_'+policy+'_f1'])
 z['nn_joint_cutoffs']=dict(sorted(collections.Counter(p['variants']['nearest_joint']['probability_threshold'] for p in pp).items()))
 context.append(z)
result={'source_checksums':{**checks,str(pairpath):sha(pairpath),str(snapshotpath):sha(snapshotpath)},'scope':'Six local judges, five backbones, ten datasets. Frozen original validation-selected cutoffs; reasoning-after has no validation sample. Judge-only and complete pipeline policies are distinct. All reasoning comparisons conditional on the fixed50 flip selection; no full-test reasoning performance claim. Weighted results describe this selected cohort only.','rows':rows,'original_full_test_macro_context':context}
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
def table(data,cols):
 def fmt(x):return f'{x:.2f}' if isinstance(x,float) else str(x)
 return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,0)))+'</td>' for k in cols)+'</tr>' for r in data)+'</table>'
body='<h1>Does changing the cutoff change the reasoning-after result?</h1><p>'+html.escape(result['scope'])+'</p><p>Positive means score ≥ cutoff. Each condition uses its own saved validation-selected cutoff. The same NN cutoff is applied to the original and reasoning-after NN probabilities. Tuned pipeline additionally applies each condition’s saved probe referral cutoff. No new calls or fitting.</p>'
for cohort,title in [('same_collected_cohort','Same collected cases: effect of asking for an explanation'),('covered_policy_flips','Subset that remains a control-versus-NN flip under each policy')]:
 body+='<h2>'+title+'</h2>'+table([r for r in rows if r['cohort']==cohort and not r['policy'].startswith('common_')],['judge','policy','n','corrected','undone','net','delta_accuracy_pp','weighted_delta_accuracy_pp'])
body+='<h2>Original flip membership and available explanation coverage</h2>'+table([r for r in rows if r['cohort']=='original'],['judge','policy','flips','helpful','harmful','overlap_fixed50','covered','coverage_pct'])
body+='<h2>Shared cutoff sweep on the same collected cases (exploratory)</h2>'+table([r for r in rows if r['policy'].startswith('common_')],['judge','policy','n','corrected','undone','net','delta_accuracy_pp'])
body+='<h2>Original full-test estimated F1 context, averaged over 50 cells per judge</h2>'+table(context,['judge','baseline_fixed50_f1','nearest_fixed50_f1','fixed50_nn_gain_pp','baseline_joint_f1','nearest_joint_f1','joint_nn_gain_pp'])
body+='<p>These are descriptive counts, not causal evidence from expressed explanations. The raw fixed50-selected cases are not representative of the full test population. Tuned-only flips without explanations cannot be evaluated for this prompt intervention. A zero judge cutoff makes every valid score positive; the pipeline then reduces to its probe referral rule.</p>'
(OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Threshold sensitivity</title><style>body{font:16px system-ui;margin:35px;max-width:1450px}table{border-collapse:collapse;margin:20px 0;font-size:13px}td,th{border:1px solid #ddd;padding:7px;text-align:right}th{background:#eef2f5}h2{margin-top:40px}</style>'+body)
for r in rows:
 if r['cohort']=='same_collected_cohort' and r['policy'] in ['fixed50_judge','tuned_judge','tuned_pipeline']:print({k:r.get(k,0) for k in ['judge','policy','n','corrected','undone','net','weighted_delta_accuracy_pp']})
print('ALL COVERAGE',*[r for r in rows if r['judge']=='ALL' and r['cohort']=='original'],sep='\n')
