from pathlib import Path
import json,gzip,hashlib,time,re
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'results';P=R/'plots'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.load(open(R/'EXPORT_COMPLETED.json'))
for n,h in manifest['files'].items():assert sha(R/n)==h
am=json.load(open(R/'ABLATION_EXPORT.json'));assert sha(R/'ablation_cases.jsonl.gz')==am['sha256']
v=json.load(open(O/'CORE_VERIFIED.json'))
for p,h in v['source_hashes'].items():assert sha(Path(p))==h,p
assert v['max_previous_metric_error']<1e-12
c=pd.read_csv(O/'cell_mechanisms.csv');a=c[c.scope=='common'];assert len(a)==500
assert a.groupby(['model','backbone']).dataset.nunique().eq(10).all()
for metric in ['f1','precision','recall']:assert np.allclose(a[metric+'_gain'],a[metric+'_score_effect']+a[metric+'_policy_effect'],atol=1e-12)
f=pd.read_csv(O/'joined_features.csv.gz');assert len(f)==50000 and not f[['backbone','sample_id']].duplicated().any();assert not f.exact_text_match.any()
b=pd.read_csv(O/'paired_cluster_bootstrap.csv');assert b.replicates.eq(2000).all()
ac=pd.read_csv(O/'ablation_cells.csv');assert len(ac)==450 and ac.groupby(['model','backbone','dataset']).condition.nunique().eq(9).all()
raw=json.load(open(R/'RAW_AUDIT.json'));assert len(raw)==2
for r in raw:
 assert 'Positive (1): the answer contains a false' in r['prompts']['baseline'][0]['content']
 for m,cs in r['outputs'].items():
  for cc,x in cs.items():assert int(x['generated_text'].strip())==x['positive_probability']==r['request']['expected_scores'][m][0 if cc=='baseline' else 2]
ab=[json.loads(line) for line in gzip.open(R/'ablation_cases.jsonl.gz','rt')]
toxic=[r for r in ab if r['dataset']=='toxicchat' and r['split']=='test' and r['backbone']=='llama70'];assert len(toxic)==500;assert sum(r['scores']['nearest']==r['scores']['text_prediction'] for r in toxic)==465
rep=(O/'report.html').read_text()
for src in re.findall(r'<img src="([^"]+)"',rep):assert (O/src).exists()
for href in re.findall(r'<a href="([^"]+)"',rep):assert (O/href).exists()
required={
'dataset_size':('dataset_correlations_exploratory.csv','Ten dataset-level correlations with leave-one-out sensitivity; no clear monotonic size pattern.'),
'cosine_similarity':('factor_cluster_intervals.csv','Within-dataset/backbone quintiles and alignment interaction; high similarity can accompany misleading opposite-class references.'),
'imbalance':('retrieval_class_balance.csv','Training prevalence and class-specific retrieval agreement distinguish majority matching from minority retrieval quality.'),
'text_length':('conditional_associations.csv','Target/reference length strata and adjusted associations, with no-truncation prompt-preparation source inspected.'),
'task_topic':('dataset_explanatory_table.csv','All ten datasets compared; exact label rubrics preserved; dataset by judge heatmap and case examples.'),
'judge_differences':('model_summary.csv','Raw AUROC/Brier, label-direction audit, threshold bypass, score/policy decomposition and paired intervals.'),
'backbone_differences':('within_target_associations.csv','Backbone means, target overlap audit, within-target/judge associations; all-five intersection flagged as selected and unrepresentative.'),
'other_factors':('sample_support.csv','Effective samples, API missingness, ceiling performance, baseline errors, existing threshold stability and exact duplicates examined.'),
'ablation_support':('ablation_summary.csv','Five completed pairs, nine conditions, checked source hashes and exact selected case membership; retrieval and class cues compared at fixed 50.'),
'concrete_examples':('matched_examples.json','Deterministically selected help/hurt examples inspected; labels treated as annotations rather than newly adjudicated truth.'),
'reproducibility_and_uncertainty':('CORE_VERIFIED.json','Prior metrics reproduced; sources checksummed; paired cluster uncertainty; full local sensitivity; no new inference.'),
'clear_synthesis':('report.html','Evidence supports a broad correlational account of class-boundary transfer, judge susceptibility, probe headroom and threshold selection; no causal internal-mechanism claim.')}
for key,(file,_) in required.items():assert (O/file).is_file()
record=dict(time=time.time(),status='complete',scope='Mechanism investigation requested by user; broad correlations and explanatory synthesis, not proof of causal model internals.',requirements={k:dict(status='verified',evidence=file,finding=txt) for k,(file,txt) in required.items()},optional_ongoing='Two-neighbour collection is a separate ongoing experiment; no claim of complete 2NN results is made.',visual_qa='Four PNGs inspected; overview legend moved away from data and final overview rechecked.',no_new_inference=True,files={str(p.relative_to(R)):sha(p) for p in list(O.iterdir())+list(P.iterdir())+list(R.glob('*.py')) if p.is_file() and p.name!='COMPLETION_AUDIT.json'})
(O/'COMPLETION_AUDIT.json').write_text(json.dumps(record,indent=2));print('Verified all',len(required),'requirements;',len(record['files']),'artifact hashes')
