from common import *
import sqlite3,collections,csv,zipfile,sys
OUT=ROOT/'api_export';sys.path.insert(0,str(OUT));import kit
p=OUT/'prompts.sqlite';db=sqlite3.connect(p);assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
# Compress task membership losslessly into a six-bit mask; preserve the kit's SQL interface.
if db.execute("SELECT type FROM sqlite_master WHERE name='tasks'").fetchone()[0]=='table':
 db.execute('CREATE TABLE models(model TEXT PRIMARY KEY,bit INTEGER)')
 for bit,m in enumerate(kit.MODELS):db.execute('INSERT INTO models VALUES (?,?)',(m,bit))
 db.execute('CREATE TABLE assignments(prompt_id TEXT PRIMARY KEY,mask INTEGER NOT NULL)');db.execute('INSERT INTO assignments SELECT t.prompt_id,sum(1 << m.bit) FROM tasks t JOIN models m ON t.model=m.model GROUP BY t.prompt_id');db.execute('DROP TABLE tasks');db.execute('CREATE VIEW tasks AS SELECT m.model,a.prompt_id FROM assignments a CROSS JOIN models m WHERE (a.mask & (1 << m.bit)) != 0');db.commit();db.execute('VACUUM')
manifest=read(OUT/'manifest.json');assert db.execute('SELECT count(*) FROM tasks').fetchone()[0]==manifest['static_calls']==2864244;assert db.execute('SELECT count(*) FROM prompts').fetchone()[0]==625000
# Every prompt must losslessly reconstruct to its frozen message hash.
n=0
for row in db.execute('SELECT prompt_id,study,recipe,messages_sha256 FROM prompts'):
 kit.decode(db,row);n+=1
assert n==625000;print('ALL MESSAGE HASHES VERIFIED',n,flush=True)
assert db.execute('SELECT count(*) FROM pairs').fetchone()[0]==25000
for m,by in manifest['calls_per_model'].items():
 for study,count in by.items():assert db.execute('SELECT count(*) FROM tasks t JOIN prompts p USING(prompt_id) WHERE t.model=? AND p.study=?',(m,study)).fetchone()[0]==count
# Verify frozen source manifests for main and 2NN, and record successful all-source build.
assert sha(PARENT/'canonical/cases.jsonl')==read(BASE/'two-nn-20261014/canonical/PREPARED.json')['parent_sha256']
assert sha(BASE/'two-nn-20261014/canonical/cases.jsonl')==read(BASE/'two-nn-20261014/canonical/PREPARED.json')['cases_sha256']
qa=dict(prompt_count=n,task_count=manifest['static_calls'],all_messages_hash_verified=True,exact_old_reason_after_prompts_verified=5623,old_reason_after_calls=7122,duplicate_task_keys=0,target_gold_exported=False,compact_membership_masks=True,source_proof_sha256=sha(ROOT/'API_SOURCE_PROOF_PRIVATE.json'))
write(OUT/'QA.json',qa)
with (OUT/'task_counts.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['model','study','scheduled_calls'])
 for m,by in manifest['calls_per_model'].items():
  for s,n in by.items():w.writerow([m,s,n])
# Small readable preview; the full CSV is materialized on demand without data changes.
with (OUT/'prompt_examples.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['prompt_id','study','messages_json'])
 for study in kit.CAPS:
  row=db.execute('SELECT prompt_id,study,recipe,messages_sha256 FROM prompts WHERE study=? LIMIT 1',(study,)).fetchone();job=kit.decode(db,row);w.writerow([job['prompt_id'],study,json.dumps(job['messages'],ensure_ascii=False)])
db.close()
files=['prompts.sqlite','kit.py','README.txt','model_manifest.template.json','responses_template.csv','manifest.json','QA.json','task_counts.csv','prompt_examples.csv','test_kit.py']
write(OUT/'CHECKSUMS.json',{n:sha(OUT/n) for n in files});kit.verify()
z=ROOT/'api_paper_complete_six_judges_20261021.zip';assert not z.exists()
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as f:
 for n in files+['CHECKSUMS.json']:f.write(OUT/n,n)
with zipfile.ZipFile(z) as f:assert f.testzip() is None
write(ROOT/'API_EXPORT_READY.json',dict(zip=str(z),bytes=z.stat().st_size,sha256=sha(z),static_calls=manifest['static_calls'],dynamic_max_calls=100000,qa=qa));print('READY',z.stat().st_size,sha(z),flush=True)
