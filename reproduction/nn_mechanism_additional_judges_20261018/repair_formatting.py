"""Auditable parser-only recovery; retains all original attempts and commits."""
import json,hashlib,time,ast,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
plan=read(R/'QUEUE_PLAN.json')
for loc in plan['children']:
 c=Path(loc);s=(c/'study.py').read_text();prep=read(c/'PREPARED.json')
 assert sha(c/'study.py')==prep['script_sha256']
 s=s.replace("assert sha(Path(__file__))==prep['script_sha256']","assert sha(R/'study.py')==prep['script_sha256'];assert sha(Path(__file__))==read(R/'PARSER_REPAIR.json')['runner_sha256']")
 s=s.replace("labels=json.loads(g['text']);assert isinstance(labels,dict) and set(labels)==set(FIELDS)","text=g['text'].strip();wrapped=False\n   import re\n   match=re.fullmatch(r'```(?:json)?\\s*\\n(.*?)\\n```',text,flags=re.DOTALL)\n   if match:text=match.group(1);wrapped=True\n   labels=json.loads(text);assert isinstance(labels,dict) and set(labels)==set(FIELDS)")
 s=s.replace("status=status,parse_error=error,raw_response", "status=status,markdown_wrapper=wrapped,parse_error=error,raw_response")
 s=s.replace("   rr=saved['records']", "   assert saved['attempt_sha256']==sha(attempt)\n   if raw.exists():\n    payload=read(raw);rr=parse(payload['responses'],chunk,batch,payload['seconds'])\n    derived=R/'normalized_v2'/committed.name\n    if not derived.exists():write(derived,dict(records=rr,original_commit_sha256=sha(committed),raw_sha256=sha(raw),parser_repair_sha256=sha(R/'PARSER_REPAIR.json')))\n   else:rr=saved['records']")
 s=s.replace("R/'FIRST_BATCH_CHECK.json'", "R/'FIRST_BATCH_CHECK_V2.json'")
 s=s.replace("write(R/'FULL_LAUNCHED.json'", "write(R/'FULL_LAUNCHED_V2.json'")
 s=s.replace("checks={str(p.relative_to(R)):sha(p) for p in (R/'batches').glob('*.json')};", "checks={str(p.relative_to(R)):sha(p) for d in ['batches','normalized_v2'] for p in (R/d).glob('*.json')};")
 ast.parse(s)
 dest=c/'run_v2.py';assert not dest.exists();dest.write_text(s)
 write(c/'PARSER_REPAIR.json',dict(time=time.time(),runner_sha256=sha(dest),original_runner_sha256=sha(c/'study.py'),policy='Accept a single entire-response Markdown JSON fence, then require the identical strict two-key schema. Preserve original raw text, invalid commits, and wrapper flag. Reparse saved outputs only; do not regenerate attempted calls. Inputs and inference configuration unchanged. Same parsing for both additional judges.',original_prepared_sha256=sha(c/'PREPARED.json')))
 # Exercise actual parser against saved first batch and reject non-wrapper extra prose.
 import importlib.util
 spec=importlib.util.spec_from_file_location('runner_'+c.name,dest);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if (c/'batches/000000.raw.json').exists():
  raw=read(c/'batches/000000.raw.json');units=list(m.jl(c/'units.jsonl'))[:16]
  rr=m.parse(raw['responses'],units,0,raw['seconds']);assert len(rr)==16 and all(x['status']=='valid' and x['markdown_wrapper'] for x in rr)
  write(c/'REPARSE_CHECK.json',dict(n=16,valid=16,raw_sha256=sha(c/'batches/000000.raw.json'),no_calls=True))
 payload=[dict(unit_id='x',outputs=[dict(text='extra prose {"labelled_similarity":"yes","important_contrast":"no"}',finish_reason='stop',token_ids=[])])]
 assert m.parse(payload,[dict(unit_id='x')],0,0)[0]['status']=='invalid'
print('Parser repair prepared and checked; all original files and inference settings preserved.')
