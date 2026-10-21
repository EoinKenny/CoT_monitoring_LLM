from pathlib import Path
import os,json,hashlib,subprocess,sys,datetime
import numpy as np
from missing_strata import select_supplement
R=Path(os.environ.get('EXPERIMENT_ROOT','/home/ubuntu/experiments/gemma31b-backbone-20261003'));S=R/'supplement'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results=json.loads((R/'probe_results.json').read_text());original=json.loads((R/'private.json').read_text());population=[];sources={}
for ds,res in results.items():
 mp=R/f'{ds}_metadata.json';pp=R/f'{ds}_predictions.npz';rows=json.loads(mp.read_text());p=np.load(pp)
 assert np.array_equal(p['labels'],[x['label'] for x in rows])
 candidates=[dict(sample_id=f'{ds}-{i}',dataset=ds,index=i,true_label=x['label'],probe_probability=float(p['probabilities'][i])) for i,x in enumerate(rows) if x['split']=='test' and p['probabilities'][i]>=res['threshold']]
 assert sum(x['true_label']==0 for x in candidates)==res['metrics']['test']['fp']
 assert sum(x['true_label']==1 for x in candidates)==res['metrics']['test']['tp']
 population.extend(candidates);sources[mp.name]=sha(mp);sources[pp.name]=sha(pp)
selected,strata=select_supplement(population,original)
frozen=dict(rule='missing-stratum-census-v1',original_sample_sha256=sha(R/'private.json'),sample_manifest_sha256=sha(R/'SAMPLE_FROZEN.json'),source_hashes=sources,strata=strata,rows=selected,extra_judgments=len(selected)*15)
manifest=R/'SUPPLEMENT_MANIFEST.json'
if manifest.exists():
 previous=json.loads(manifest.read_text());assert previous['frozen']==frozen,'Supplement inputs changed; refusing resampling'
else:
 with manifest.open('x') as f:json.dump(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen=frozen,amendment='Post hoc for this Gemma run after missing-stratum scoring failure; fixed before supplemental judge outcomes. Original uniform sample remains unchanged. Census missing nonempty gold-class strata, same cases across all conditions. Sampled-only metrics exclude supplements; cascade class acceptance uses census exactly; census strata are not bootstrapped.'),f,indent=2)
print(json.dumps(dict(supplement_cases=len(selected),strata=strata,extra_judgments=len(selected)*15)),flush=True)
if not selected:sys.exit(0)
S.mkdir(exist_ok=True)
if not (S/'SUPPLEMENT_ROWS.json').exists():(S/'SUPPLEMENT_ROWS.json').write_text(json.dumps(selected))
else:assert json.loads((S/'SUPPLEMENT_ROWS.json').read_text())==selected
for name in ['backbone.json','judge_rubrics.json','COMPLETED']+[f'{ds}_{suffix}' for ds in results for suffix in ['metadata.json','features.npy','predictions.npz','probe.joblib']]:
 dest=S/name
 if not dest.exists():dest.symlink_to(R/name)
if not (S/'results.json').exists():(S/'results.json').symlink_to(R/'probe_results.json')
env=dict(os.environ,EXPERIMENT_ROOT=str(S))
def call(script,args=[]):
 log=S/(script.replace('.py','')+('_'+'_'.join(args) if args else '')+'.log')
 with log.open('x') as f:subprocess.run([sys.executable,str(R/script)]+args,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
if not (S/'judge_inputs/EXTRACTION_COMPLETED').exists():call('explanations.py')
for model in ['qwen','gemma','llama','glm','olmo']:
 if not (S/(model+'_PROMPTS_VERIFIED')).exists():
  call('prepare_judges.py',[model]);(S/(model+'_PROMPTS_VERIFIED')).write_text('complete')
for c in ['baseline','saliency','nearest']:
 cfg=[json.loads((S/'judges'/f'{m}_{c}'/'approved_config.json').read_text()) for m in ['qwen','gemma','llama','glm','olmo']]
 assert len({x['semantic_sha256'] for x in cfg})==1
 assert all(x['n']==len(selected) for x in cfg)
(S/'PREPARED').write_text(json.dumps(dict(manifest_sha256=sha(manifest),cases=len(selected))))
