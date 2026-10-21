from pathlib import Path
import sys,sqlite3,json,lzma,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'api_export';OUT=ROOT/'api_export_compact'
sys.path.insert(0,str(OUT));from restore import logical_checks,encoded
db=sqlite3.connect('file:'+str(SRC/'prompts.sqlite')+'?mode=ro',uri=True)

(OUT/'CATALOG_LOGICAL.json').write_text(json.dumps(logical_checks(db),indent=2))
schema=db.execute('SELECT type,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid').fetchall()
cases=[r[0] for r in db.execute('SELECT DISTINCT case_key FROM prompts ORDER BY case_key')];ci={x:i for i,x in enumerate(cases)}
masks=dict(db.execute('SELECT * FROM assignments'));pids={}
identities=json.loads((ROOT/'compact_case_identity.json').read_text())
parents={}
for case,a,n,after,first in db.execute('SELECT * FROM pairs'):
 parents[after]=n;parents[first]=n
def h(s):return hashlib.sha256(s.encode()).hexdigest()
with lzma.open(OUT/'catalog_final.jsonl.xz','wb',preset=9) as f:
 def emit(x):f.write(encoded(x)+b'\n')
 emit({'format':'logical-catalog-v2','tables':[sql for t,sql in schema if t=='table'],'post_schema':[sql for t,sql in schema if t!='table']})
 for r in db.execute('SELECT * FROM strings ORDER BY text'):emit(['s',r])
 emit(['c',[identities.get(x,x) for x in cases]]);print('Strings packed',flush=True)
 for i,r in enumerate(db.execute('SELECT * FROM prompts ORDER BY rowid')):
  pid,study,recipe,mh,case,split,condition=r;pids[pid]=i
  packed_pid=pid
  if study=='ablation':
   b,sample=identities[case];assert pid=='apiabl_'+h(b+':nnabl_'+h(json.dumps([sample,condition,mh],separators=(',',':'))));packed_pid=None
  elif study in ['reason_after','reason_first']:
   parent=parents[pid];assert pid==('apir_' if study=='reason_after' else 'apif_')+h(parent+':'+mh);packed_pid=pids[parent]
  emit(['p',[packed_pid,study,recipe,ci[case],split,condition,masks.get(pid)]])
  if (i+1)%100000==0:print('Packed',i+1,flush=True)
 for r in db.execute('SELECT * FROM pairs ORDER BY case_key'):emit(['r',[ci[r[0]]]+[pids[x] for x in r[1:]]])
 for r in db.execute('SELECT * FROM metadata ORDER BY key'):emit(['m',r])
 for r in db.execute('SELECT * FROM models ORDER BY model'):emit(['j',r])
print('Packed bytes',(OUT/'catalog_final.jsonl.xz').stat().st_size,flush=True)
