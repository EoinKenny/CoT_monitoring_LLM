"""Independent identity, source, arithmetic and report-link audit."""
from pathlib import Path
import json,gzip,hashlib,re,time
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rec=json.load(open(O/'ANALYSIS_COMPLETED.json'))
for p,h in rec['sources'].items():assert sha(Path(p))==h,p
f=pd.read_csv(P/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz');scores=np.load(P/'api_results_20261015/results/scores.npz');design=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'));d=pd.DataFrame([dict(backbone=b,**r) for b,rs in design['samples'].items() for r in rs]);z=d.iloc[f.source_index].reset_index(drop=True)
assert len(f)==25000 and f.split.eq('test').all() and f.unit_id.is_unique and not f.duplicated(['backbone','sample_id']).any()
for key in ['backbone','sample_id','dataset','true_label','split']:assert np.array_equal(f[key].to_numpy(),z[key].to_numpy()),key
serialization_errors={}
for key in ['probe_probability','inclusion_probability']:
 err=float(np.max(abs(f[key]-z[key])));assert err<1e-14;serialization_errors[key]=err
assert np.array_equal(f.probe_probability.ge(.5),z.probe_probability.ge(.5))
th=pd.DataFrame(json.load(open(P/'anchor_transfer_20261018/results/thresholds.json')));th=th[th.scope.ne('common_valid')];ff=f.copy();ff['original_probability']=z.probe_probability;aa=ff.merge(th,on=['backbone','dataset']);assert np.array_equal(aa.probe_probability.ge(aa.probe_threshold),aa.original_probability.ge(aa.probe_threshold))
assert np.allclose(f.weight,1/f.inclusion_probability)
frames=[]
for model in scores.files:
 vals=scores[model][f.source_index];valid=(vals>=0).all(axis=1);h=f[valid].copy();vals=vals[valid];assert np.isfinite(vals).all() and (vals<=100).all()
 h['model']=model;h['control']=vals[:,0];h['nn']=vals[:,2];h['w']=h.weight/h.groupby(['backbone','dataset']).weight.transform('sum');h['cell']=model+'|'+h.backbone+'|'+h.dataset
 h['gain']=((h.nn>=50)==h.true_label).astype(int)-((h.control>=50)==h.true_label).astype(int);h['harm']=h.gain.eq(-1);h['help']=h.gain.eq(1);h['toward']=(2*h.nearest_label-1)*(h.nn-h.control)>0
 frames.append(h)
a=pd.concat(frames);assert len(a)==240524 and a.cell.nunique()==500
s=pd.read_csv(O/'summary.csv');checks=0
for cohort,h in [('all10',a),('toxicity4',a[a.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])])]:
 for contrast in ['yes','no']:
  v=h[h.majority_important_contrast.eq(contrast)];row=s.query("policy=='judge50_direct' and cohort==@cohort and annotator=='majority' and similarity=='all' and contrast==@contrast").iloc[0]
  assert len(v)==row.n and np.isclose(np.average(v.gain,weights=v.w),row.gain)
  assert np.isclose(np.average(v.nearest_label.eq(v.true_label),weights=v.w),row.aligned)
  if contrast=='yes':
   harm=v[v.harm];rr=pd.read_csv(O/'contrast_flip_decomposition.csv').query("cohort==@cohort and scope=='all_contrast' and outcome=='harmed' and direction=='toward'").iloc[0];assert np.isclose(np.average(harm.toward,weights=harm.w),rr.fraction)
  checks+=1
# Verify all exact displayed candidates against original text hashes and original scores.
ex=json.load(open(O/'audit_examples.json'));assert len(ex)==32
fm={(r.backbone,r.sample_id):r for _,r in f.iterrows()}
for e in ex:
 r=e['source'];rr=fm[r['backbone'],r['sample_id']]
 for prefix,key in [('target','target_messages'),('neighbor','neighbor_messages')]:
  text='\n'.join(str(m.get('content','')) for m in r[key]);assert hashlib.sha256(text.encode()).hexdigest()==r[prefix+'_content_hash']
 assert r['true_label']==rr.true_label and r['nearest_label']==rr.nearest_label
 for j in e['judge_outcomes']:
  val=scores[j['judge']][rr.source_index];assert j['control']==val[0] and j['nearest']==val[2]
notes=json.load(open(R/'qualitative_notes.json'));assert len(notes['examples'])==8 and {x['audit_id'] for x in notes['examples']}<={e['audit_id'] for e in ex}
for name,col in [('regressions.csv','fe_converged'),('risk_regressions.csv','converged')]:
 t=pd.read_csv(O/name);assert t[col].all() and np.isfinite(t[['estimate','lo','hi']]).all().all();assert ((t.lo<=t.estimate)&(t.estimate<=t.hi)).all()
 assert (R/'numerical_audit'/('original_'+name)).exists()
for filename,est in [('second_nn_interaction.csv','difference'),('second_nn_sensitivity.csv','delta'),('contrast_component_ablations.csv','delta_vs_full')]:
 t=pd.read_csv(O/filename);assert np.isfinite(t[[est,'lo','hi']]).all().all();assert (t.hi>=t.lo).all()
# Independent cluster multiplier check of the flagship 83.2% estimate.
g=a[a.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])&a.majority_important_contrast.eq('yes')&a.harm].copy();g['wy']=g.w*g.toward;cl=g.groupby('cluster')[['w','wy']].sum();rng=np.random.default_rng(20261020);est=[]
for i in range(2000):
 mult=rng.exponential(size=len(cl));est.append(np.dot(mult,cl.wy)/np.dot(mult,cl.w))
ci=np.quantile(est,[.025,.975]);assert ci[0]>.75 and ci[1]<.90
text=(O/'report.html').read_text();local=[]
for target in re.findall(r'(?:src|href)="([^"]+)"',text):
 if target.startswith(('http','#')):continue
 if target=='VERIFIED.json':continue
 assert (O/target).exists(),target;local.append(target)
figures=list(O.glob('*.png'));assert len(figures)==7
for p in figures:assert p.stat().st_size>10000 and p.with_suffix('.pdf').exists()
assert '83.2%' in text and 'No new model calls' in text
# Current hashes, including scripts, for reproducibility.
source_files=[P/'nn_mechanisms_20261015/features.jsonl.gz',P/'nn_reason_flips_20261017/FINAL_ANALYSIS_SNAPSHOT.json.gz',P/'nn_mechanisms_20261015/results/ablation_case_effects.csv.gz',P/'anchor_transfer_20261018/results/thresholds.json',P/'two_nn_20261014/export_verified_20261016/nearest2_scores.jsonl.gz',P/'nn_mechanism_two_labels_20261018/prompt_template.txt',P/'nn_mechanism_agreement_20261019/results/agreement.csv']
receipt=dict(time=time.time(),original_source_hashes_unchanged=True,design_membership_labels_exact=True,numeric_serialization_max_errors=serialization_errors,all_frozen_referral_decisions_exact=True,weights_checked_against_inclusion_probabilities=True,contexts=len(f),target_clusters=f.cluster.nunique(),judge_context_rows=len(a),cells=a.cell.nunique(),recomputed_primary_subgroup_checks=checks,audit_candidates_verified=len(ex),closely_reviewed_examples=8,all_regression_projections_converged=True,flagship_harm_toward_estimate=float(np.average(g.toward,weights=g.w)),flagship_cluster_multiplier_95_interval=ci.tolist(),bootstrap_replicates=2000,local_report_links_verified=len(local),figure_pairs_verified=len(figures),new_model_calls=0,geometry_limitation='Remote SSH timed out; no exact probe coefficient norms or second-neighbour class metadata recovered. Saved score-derived within-probe ranks used instead.',sources={str(p):sha(p) for p in source_files},files={str(p.relative_to(R)):sha(p) for p in list(R.glob('*.py'))+list(O.glob('*.csv'))+list(O.glob('*.png'))+list(O.glob('*.pdf'))+[O/'report.html',O/'audit_examples.json',O/'audit_second_nn_scores.json',O/'CLAIMS.json',R/'qualitative_notes.json',R/'literature.json']})
(O/'VERIFIED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ['sources','files']},indent=2))
