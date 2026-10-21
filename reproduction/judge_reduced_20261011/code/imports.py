"""Verify saved generations at their original immutable source; never re-query."""
from common import *
def verify_imports(path):
 path=Path(path)
 if not path.exists():return []
 data=read(path);out=[]
 for item in data['batches']:
  src=Path(item['source_batch']);raw=src.with_suffix('.raw.json');cfg=Path(item['source_config'])
  assert sha(src)==item['source_batch_sha256'] and sha(raw)==item['source_raw_sha256'] and sha(cfg)==item['source_config_sha256']
  doc=read(src)
  assert doc['raw_sha256']==sha(raw) and doc['config_sha256']==sha(cfg) and doc['records_sha256']==digest(json.dumps(doc['records'],sort_keys=True))
  wanted=set(item['selected_ids']);rr=[r for r in doc['records'] if r['sample_id'] in wanted]
  assert len(rr)==len(wanted)
  out+=rr
 lookup={r['sample_id']:r for r in out}
 assert len(lookup)==len(out)==len(data['ordered_ids'])
 rows=[lookup[i] for i in data['ordered_ids']]
 assert digest(json.dumps(rows,sort_keys=True))==data['records_sha256']
 return rows
def create_imports(source,dest,wanted):
 source=Path(source);dest=Path(dest);cfg=source/'run/config.json'
 if not cfg.exists():return []
 items=[];rr=[]
 for p in sorted((source/'run/batches').glob('*.json')):
  if p.name.endswith(('.attempt.json','.raw.json')):continue
  d=read(p);selected=[r for r in d['records'] if r['sample_id'] in wanted]
  if not selected:continue
  assert d['raw_sha256']==sha(p.with_suffix('.raw.json')) and d['config_sha256']==sha(cfg)
  assert d['records_sha256']==digest(json.dumps(d['records'],sort_keys=True))
  items.append(dict(source_batch=str(p),source_batch_sha256=sha(p),source_raw_sha256=d['raw_sha256'],source_config=str(cfg),source_config_sha256=sha(cfg),selected_ids=[r['sample_id'] for r in selected]))
  rr+=selected
 lookup={r['sample_id']:r for r in rr};assert len(lookup)==len(rr)
 ids=[r['sample_id'] for r in jl(dest/'prompts.jsonl') if r['sample_id'] in lookup]
 ordered=[lookup[i] for i in ids]
 write(dest/'run/IMPORTED.json',dict(time=time.time(),reason='Retrospective outcome-blind projection of preserved committed outputs; original batch context retained',batches=items,ordered_ids=ids,records_sha256=digest(json.dumps(ordered,sort_keys=True))))
 assert verify_imports(dest/'run/IMPORTED.json')==ordered
 return ordered
