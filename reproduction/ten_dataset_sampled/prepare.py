from pathlib import Path
import json,hashlib,shutil,platform,subprocess
B=Path('/home/ubuntu/experiments/ten-dataset-full-20261002')
R=B.parent/'ten-dataset-sampled-20261002'
SEED='20261002';CAP=500
R.mkdir(exist_ok=True)
assert not (R/'SAMPLE_FROZEN.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rank(r):return hashlib.sha256((SEED+'\0'+r['dataset']+'\0'+str(r['sample_id'])).encode()).hexdigest()
rows=json.loads((B/'judge_inputs/private.json').read_text())
assert len({r['sample_id'] for r in rows})==len(rows)
datasets=sorted({r['dataset'] for r in rows});selected=[];counts={}
for ds in datasets:
 pool=[r for r in rows if r['dataset']==ds];part=sorted(pool,key=lambda r:(rank(r),str(r['sample_id'])))[:CAP]
 selected.extend(part);counts[ds]={'population_flags':len(pool),'sampled_flags':len(part),'inclusion_probability':len(part)/len(pool)}
ids={r['sample_id'] for r in selected}
manifest=dict(seed=SEED,cap=CAP,algorithm='Sort each dataset by SHA256(seed + NUL + dataset + NUL + sample_id), tie by sample_id; take first min(500,N). Independent of labels and judge outcomes.',counts=counts,sample_ids=[r['sample_id'] for r in selected],source_private_sha256=sha(B/'judge_inputs/private.json'))
(R/'private.json').write_text(json.dumps(selected))
(R/'SAMPLE_FROZEN.json').write_text(json.dumps(manifest,indent=2))
(R/'weighted_results.json').write_bytes((B/'judge_inputs/weighted_results.json').read_bytes())
reused_counts={};semantics={}
for model in ['qwen','gemma','llama']:
 for condition in ['baseline','saliency','nearest']:
  name=model+'_'+condition;src=B/'judges'/name;out=R/'judges'/name;out.mkdir(parents=True)
  prompts=[json.loads(l) for l in (src/'prompts.jsonl').read_text().splitlines()];byid={p['sample_id']:p for p in prompts};chosen=[byid[r['sample_id']] for r in selected]
  assert set(p['sample_id'] for p in chosen)==ids
  (out/'prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in chosen))
  sem=hashlib.sha256(''.join(p['instruction_sha256'] for p in chosen).encode()).hexdigest();semantics.setdefault(condition,[]).append(sem)
  cfg=json.loads((src/'approved_config.json').read_text());cfg.update(n=len(chosen),source_config_sha256=sha(src/'approved_config.json'),source_root=str(src),semantic_sha256=sem,prompt_sha256=sha(out/'prompts.jsonl'),sample_manifest_sha256=sha(R/'SAMPLE_FROZEN.json'),max_input_tokens=max(p['input_tokens'] for p in chosen),total_input_tokens=sum(p['input_tokens'] for p in chosen),protocol='Frozen uniform flag sample; full-test probe; estimated cascade; unchanged source prompts')
  # Reuse only durable valid records with exact rendered prompt, tokens and semantic identity.
  reuse=[];prior=src/'run/judgments.jsonl'
  if prior.exists():
   prior_sha=sha(prior)
   oldprompts={p['sample_id']:p for p in [json.loads(l) for l in (src/'run/prompts.jsonl').read_text().splitlines()]}
   priorrows=[json.loads(l) for l in prior.read_text().splitlines()];assert len({r['sample_id'] for r in priorrows})==len(priorrows)
   for row in priorrows:
    sid=row['sample_id']
    if sid not in ids:continue
    assert oldprompts[sid]==byid[sid] and row['prediction'] in [0,1] and row['dataset']==byid[sid]['dataset']
    row=dict(row,reused_from=str(prior),source_judgments_sha256=prior_sha,matched_prompt_sha256=hashlib.sha256(json.dumps(byid[sid],sort_keys=True).encode()).hexdigest());reuse.append(row)
  cfg['reused']=len(reuse);reused_counts[name]=len(reuse)
  (out/'reused.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reuse))
  (out/'approved_config.json').write_text(json.dumps(cfg,indent=2))
  shutil.copyfile(src/'model_path.txt',out/'model_path.txt')
assert all(len(set(s))==1 for s in semantics.values())
# Archive split membership and exact source scripts/checksums; private data stay on instance.
archive=R/'provenance';archive.mkdir()
files=list(B.glob('*.py'))+list(B.glob('*_metadata.json'))+list(B.glob('*_probe.joblib'))+[B/n for n in ['downloads.json','results.json','VERIFIED.json','judge_rubrics.json','README.md'] if (B/n).exists()]
checks={}
for p in files:
 shutil.copyfile(p,archive/p.name);checks[str(p)]=sha(p)
(archive/'source_checksums.json').write_text(json.dumps(checks,indent=2))
import importlib.metadata
(archive/'cpu_packages.json').write_text(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),indent=2))
(R/'PREPARED.json').write_text(json.dumps(dict(n=len(selected),counts=counts,reused=reused_counts,source=str(B)),indent=2))
print((R/'PREPARED.json').read_text())
