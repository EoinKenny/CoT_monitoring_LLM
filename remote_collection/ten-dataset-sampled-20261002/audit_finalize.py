from pathlib import Path
import json,hashlib,importlib.metadata,collections,platform,subprocess
R=Path('/home/ubuntu/experiments/ten-dataset-sampled-20261002');B=R.parent/'ten-dataset-full-20261002'
manifest=json.loads((R/'SAMPLE_FROZEN.json').read_text());source=json.loads((B/'judge_inputs/private.json').read_text());rows=json.loads((R/'private.json').read_text())
def rank(r):return hashlib.sha256(('20261002\0'+r['dataset']+'\0'+str(r['sample_id'])).encode()).hexdigest()
expected=[]
for ds in sorted(manifest['counts']):expected.extend(sorted([r for r in source if r['dataset']==ds],key=lambda r:(rank(r),str(r['sample_id'])))[:500])
assert rows==expected and [r['sample_id'] for r in rows]==manifest['sample_ids']
assert len(rows)==4573
counts={ds:dict(collections.Counter(str(r['true_label']) for r in rows if r['dataset']==ds)) for ds in manifest['counts']}
assert all(int(c.get('0',0))>0 and int(c.get('1',0))>0 for c in counts.values())
reused={};sem={}
for model in ['qwen','gemma','llama']:
 for cond in ['baseline','saliency','nearest']:
  name=model+'_'+cond;p=R/'judges'/name;cfg=json.loads((p/'approved_config.json').read_text());prompts=[json.loads(l) for l in (p/'prompts.jsonl').read_text().splitlines()]
  assert [r['sample_id'] for r in prompts]==manifest['sample_ids'] and len(prompts)==cfg['n']
  assert hashlib.sha256((p/'prompts.jsonl').read_bytes()).hexdigest()==cfg['prompt_sha256']
  reused[name]=cfg['reused'];sem.setdefault(cond,[]).append(cfg['semantic_sha256'])
assert all(len(set(v))==1 for v in sem.values())
(R/'provenance/cpu_packages.json').write_text(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),indent=2))
(R/'provenance/host.json').write_text(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)),indent=2))
(R/'PREPARATION_RECOVERY.json').write_text(json.dumps(dict(reason='Initial preparation completed sample, prompts and source archives, but failed recording package versions because pip is absent in CPU environment. Recovered using importlib.metadata; audited existing files without resampling or regenerating prompts.',sample_counts=counts),indent=2))
(R/'PREPARED.json').write_text(json.dumps(dict(n=len(rows),counts=manifest['counts'],sample_classes=counts,reused=reused,source=str(B)),indent=2))
(B/'SUPERSEDED_BY_SAMPLED.json').write_text(json.dumps(dict(new_root=str(R),reason='User requested at most 500 reproducibly sampled flags per dataset and approximately 12 hour budget; original supervisor and current judge stopped; durable valid completed judgments retained.'),indent=2))
print((R/'PREPARED.json').read_text())
