import pathlib,json,hashlib,sys
p=pathlib.Path('/home/ubuntu/experiments/paper-complete-20261021');r=json.loads((p/'recovery_20261023/COLLECTION_RESOLVED.json').read_text())
for t in r['tasks']:
 if t['task']['study']!='mechanism':continue
 f=pathlib.Path(t['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==t['sha256']
 for line in f.open():
  x=json.loads(line);print(json.dumps({k:x.get(k) for k in ['judge','sample_id','prompt_id','backbone','dataset','status','value']}))
