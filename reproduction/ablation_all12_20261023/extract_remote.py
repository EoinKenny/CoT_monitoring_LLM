import pathlib,json,hashlib,csv,gzip,sys
P=pathlib.Path('/home/ubuntu/experiments/paper-complete-20261021')
w=csv.writer(sys.stdout);w.writerow(['judge','backbone','sample_id','condition','score'])
r=json.loads((P/'recovery_20261023/COLLECTION_RESOLVED.json').read_text())
for t in r['tasks']:
 if t['task']['study']!='ablation':continue
 p=pathlib.Path(t['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==t['sha256']
 for l in p.open():
  x=json.loads(l);assert x['status']=='valid';w.writerow([x['judge'],x['backbone'],x['sample_id'],x['condition'],x['value']])
for t in json.loads((P/'REUSED_COVERAGE.json').read_text())['ablations']:
 p=pathlib.Path(t['path'])/'judgments.jsonl'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==t['sha256']
 for l in p.open():
  x=json.loads(l);w.writerow([t['judge'],t['backbone'],x['sample_id'],t['condition'],x['positive_probability']])
