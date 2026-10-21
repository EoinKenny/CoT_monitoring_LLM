from pathlib import Path
import sys,sqlite3,json,lzma,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'api_export';OUT=ROOT/'api_export_compact'
sys.path.insert(0,str(OUT));from restore import logical_checks,encoded
db=sqlite3.connect('file:'+str(SRC/'prompts.sqlite')+'?mode=ro',uri=True)
for p in SRC.iterdir():
 if p.is_file() and p.name not in ['prompts.sqlite','CHECKSUMS.json']:shutil.copy2(p,OUT/p.name)
(OUT/'CATALOG_LOGICAL.json').write_text(json.dumps(logical_checks(db),indent=2))
schema=db.execute('SELECT type,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid').fetchall()
cases=[r[0] for r in db.execute('SELECT DISTINCT case_key FROM prompts ORDER BY case_key')];ci={x:i for i,x in enumerate(cases)}
masks=dict(db.execute('SELECT * FROM assignments'));pids={}
with lzma.open(OUT/'catalog.jsonl.xz','wb',preset=9) as f:
 def emit(x):f.write(encoded(x)+b'\n')
 emit({'format':'logical-catalog-v1','tables':[sql for t,sql in schema if t=='table'],'post_schema':[sql for t,sql in schema if t!='table']})
 for r in db.execute('SELECT * FROM strings ORDER BY id'):emit(['s',r])
 emit(['c',cases]);print('Strings packed',flush=True)
 for i,r in enumerate(db.execute('SELECT * FROM prompts ORDER BY rowid')):
  pid,study,recipe,mh,case,split,condition=r;pids[pid]=i
  emit(['p',[pid,study,recipe,ci[case],split,condition,masks.get(pid)]])
  if (i+1)%100000==0:print('Packed',i+1,flush=True)
 for r in db.execute('SELECT * FROM pairs ORDER BY case_key'):emit(['r',[ci[r[0]]]+[pids[x] for x in r[1:]]])
 for r in db.execute('SELECT * FROM metadata ORDER BY key'):emit(['m',r])
 for r in db.execute('SELECT * FROM models ORDER BY model'):emit(['j',r])
print('Packed bytes',(OUT/'catalog.jsonl.xz').stat().st_size,flush=True)
