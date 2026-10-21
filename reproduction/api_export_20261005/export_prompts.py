import ast,csv,json,hashlib
from pathlib import Path
R=Path('/home/ubuntu/experiments/reproducible-sampled-20261004');D=R.parent/'api-judge-export-20261005';D.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
source=R/'gemma/prepare_judges.py';tree=ast.parse(source.read_text());constants={"json":json}
for stmt in tree.body:
 if isinstance(stmt,ast.Assign) and isinstance(stmt.targets[0],ast.Name) and stmt.targets[0].id in ['common','instructions','names','positive']:constants[stmt.targets[0].id]=ast.literal_eval(stmt.value)
# Extract the established message construction, ending BEFORE any tokenizer template.
s=source.read_text();start=s.index("   ds=r['dataset'];user=");end=s.index(';text=tok.apply_chat_template',start)
body=s[start:end];build=compile('def build(r,condition,rubrics):\n'+body+'\n   return messages\n',str(source),'exec');exec(build,constants)
models=['qwen','gemma','llama','glm','olmo'];conditions=['baseline','saliency','nearest'];count=0;seen=set()
fields=['schema_version','row_id','backbone','backbone_repo','backbone_revision','dataset','condition','sample_id','source_group','system_prompt','user_prompt','messages_sha256','source_prompt_sha256','selection_sha256','probe_sha256','probe_threshold','probe_probability','gold_label','probe_prediction','stratum','selection_hash','inclusion_probability','population_weight','population_tp','population_fp','selected_tp','selected_fp','test_tp','test_fp','test_fn','test_tn','test_n']+['local_'+m+'_prediction' for m in models]
if (D/'prompts.csv').exists():
 assert not (D/'prompts.failed-partial-v2.csv').exists()
 (D/'prompts.csv').rename(D/'prompts.failed-partial-v2.csv')
with (D/'prompts.csv').open('x',newline='',encoding='utf-8') as out:
 writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
 for b in ['llama','qwen','gemma']:
  B=R/b;bb=read(B/'backbone.json');rubrics=read(B/'judge_rubrics.json');sample=read(B/'selected_flags.json');frozen=read(B/'SAMPLE_FROZEN.json');assert sha(B/'selected_flags.json')==frozen['selection_sha256'];by={r['sample_id']:r for r in sample};results=read(B/'results.json')
  ex=[json.loads(l) for l in (B/'judge_inputs/explanations.jsonl').open()];assert [r['sample_id'] for r in ex]==frozen['sample_ids']
  for c in conditions:
   previous={};judgments={}
   for m in models:
    root=B/'judges'/f'{m}_{c}';cfg=read(root/'approved_config.json');assert sha(root/'prompts.jsonl')==cfg['prompt_sha256'];previous[m]={r['sample_id']:r['instruction_sha256'] for r in map(json.loads,(root/'prompts.jsonl').open())}
    mark=read(root/'run/COMPLETED');assert sha(root/'run/judgments.jsonl')==mark['judgments_sha256'];judgments[m]={r['sample_id']:r['prediction'] for r in map(json.loads,(root/'run/judgments.jsonl').open())}
   for r in ex:
    ds=r['dataset'];sid=r['sample_id'];a=by[sid];messages=constants['build'](r,c,rubrics);assert [m['role'] for m in messages]==['system','user'];h=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest();assert all(previous[m][sid]==h for m in models)
    rowid=hashlib.sha256(json.dumps([b,ds,c,sid,h],separators=(',',':')).encode()).hexdigest();assert rowid not in seen;seen.add(rowid)
    strata=frozen['counts'][ds]['strata'];base=results[ds]['metrics']['test']
    row=dict(schema_version='probe-api-export-v1',row_id=rowid,backbone=b,backbone_repo=bb.get('repo',bb.get('model')),backbone_revision=bb['revision'],dataset=ds,condition=c,sample_id=sid,source_group=a['group'],system_prompt=messages[0]['content'],user_prompt=messages[1]['content'],messages_sha256=h,source_prompt_sha256=h,selection_sha256=frozen['selection_sha256'],probe_sha256=sha(B/f'{ds}_probe.joblib'),probe_threshold=results[ds]['threshold'],probe_probability=a['probe_probability'],gold_label=a['true_label'],probe_prediction=1,stratum=a['stratum'],selection_hash=a['selection_hash'],inclusion_probability=a['inclusion_probability'],population_weight=a['population_weight'],population_tp=strata['TP']['population'],population_fp=strata['FP']['population'],selected_tp=strata['TP']['selected'],selected_fp=strata['FP']['selected'],**{'test_'+k:base[k] for k in ['tp','fp','fn','tn','n']},**{'local_'+m+'_prediction':judgments[m][sid] for m in models})
    writer.writerow(row);count+=1
assert count==40179
print(json.dumps(dict(rows=count,sha256=sha(D/'prompts.csv'),bytes=(D/'prompts.csv').stat().st_size,all_5_local_judges_semantically_verified=True)))
