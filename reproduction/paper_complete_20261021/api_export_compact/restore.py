"""Restore the exact logical prompt catalog; standard library only, no network."""
from pathlib import Path
import hashlib,json,lzma,sqlite3,os
ROOT=Path(__file__).resolve().parent
TABLES={'strings':'id','prompts':'rowid','pairs':'case_key','metadata':'key','models':'model','assignments':'prompt_id'}
def encoded(x):return json.dumps(x,ensure_ascii=False,separators=(',',':')).encode('utf-8')
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def logical_checks(db):
 out={}
 for t,order in TABLES.items():
  h=hashlib.sha256();n=0
  for row in db.execute('SELECT * FROM '+t+' ORDER BY '+order):h.update(encoded(row)+b'\n');n+=1
  out[t]={'rows':n,'sha256':h.hexdigest()}
 return out
def verify_catalog(db):
 expected=json.loads((ROOT/'CATALOG_LOGICAL.json').read_text())
 assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
 actual=logical_checks(db)
 assert actual==expected,(actual,expected)
 assert db.execute('SELECT count(*) FROM tasks').fetchone()[0]==2864244
 return actual
def main():
 checks=json.loads((ROOT/'CHECKSUMS.json').read_text())
 for name,digest in checks.items():assert file_sha(ROOT/name)==digest,name
 target=ROOT/'prompts.sqlite'
 if target.exists():
  db=sqlite3.connect('file:'+str(target)+'?mode=ro',uri=True);verify_catalog(db);db.close()
  print('Existing catalog verified; nothing overwritten.');return
 temp=ROOT/'prompts.restoring.sqlite'
 if temp.exists():raise RuntimeError('Incomplete restoration exists. Preserve or remove prompts.restoring.sqlite before restarting.')
 db=sqlite3.connect(temp);db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
 strings={};cases=[];identities=[];pids=[]
 with lzma.open(ROOT/'catalog.jsonl.xz','rt',encoding='utf-8') as f:
  header=json.loads(next(f));assert header['format']=='logical-catalog-v2'
  for sql in header['tables']:db.execute(sql)
  for line in f:
   tag,row=json.loads(line)
   if tag=='s':db.execute('INSERT INTO strings VALUES (?,?)',row);strings[row[0]]=row[1]
   elif tag=='c':
    identities=row
    cases=[hashlib.sha256((x[0]+':'+x[1]).encode()).hexdigest() if isinstance(x,list) else x for x in row]
   elif tag=='p':
    pid,study,recipe,ci,split,condition,mask=row
    # Regenerate the redundant digest from the exact original role/content bytes.
    messages=[{'role':role,'content':''.join(strings[i] for i in ids)} for role,ids in json.loads(recipe)]
    mh=hashlib.sha256(encoded(messages)).hexdigest()
    if pid is None:
     assert study=='ablation';backbone,sample_id=identities[ci]
     old='nnabl_'+hashlib.sha256(encoded([sample_id,condition,mh])).hexdigest()
     pid='apiabl_'+hashlib.sha256((backbone+':'+old).encode()).hexdigest()
    elif isinstance(pid,int):
     assert study in ('reason_after','reason_first')
     pid=('apir_' if study=='reason_after' else 'apif_')+hashlib.sha256((pids[pid]+':'+mh).encode()).hexdigest()
    db.execute('INSERT INTO prompts VALUES (?,?,?,?,?,?,?)',(pid,study,recipe,mh,cases[ci],split,condition));pids.append(pid)
    if mask is not None:db.execute('INSERT INTO assignments VALUES (?,?)',(pid,mask))
   elif tag=='r':db.execute('INSERT INTO pairs VALUES (?,?,?,?,?)',[cases[row[0]]]+[pids[i] for i in row[1:]])
   elif tag=='m':db.execute('INSERT INTO metadata VALUES (?,?)',row)
   elif tag=='j':db.execute('INSERT INTO models VALUES (?,?)',row)
   else:raise ValueError(tag)
  for sql in header['post_schema']:db.execute(sql)
 db.commit();print('Restored catalog; verifying all rows and task assignments...',flush=True)
 verify_catalog(db);db.close();os.replace(temp,target)
 print('Verified: 625000 exact prompts, 2864244 static assignments, both reasoning orders. Run: python3 kit.py verify')
if __name__=='__main__':main()
