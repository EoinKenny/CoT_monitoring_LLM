from pathlib import Path
import os,sys,json,hashlib,gzip,csv,time,shutil
BASE=Path(os.environ.get('EXPERIMENTS_ROOT','/home/ubuntu/experiments'));OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True);ROOTS=json.loads(Path(sys.argv[2]).read_text())
FIELDS=['experiment','source_path','source_row','judge','backbone','dataset','split','sample_id','prompt_id','condition','positive_probability','prediction','status','finish_reason','raw_response','raw_record_json']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
for name in ROOTS:
 root=BASE/name;dest=OUT/(name+'.csv.gz');receipt=OUT/(name+'.VERIFIED.json')
 if receipt.exists():continue
 assert not dest.exists();tmp=dest.with_suffix('.gz.partial');assert not tmp.exists();sources=[];n=0;changes=[]
 files=[]
 for directory,dirs,fs in os.walk(root):
  dirs[:]=[d for d in dirs if not d.startswith('.venv') and d not in ['.git','__pycache__','api_export','site-packages']]
  for f in fs:
   p=Path(directory)/f
   if f in ['judgments.jsonl','responses.jsonl','annotations.jsonl','reused.jsonl']:files.append(p)
 # Freeze which active tasks use aggregate files vs committed batches before reading.
 if name=='paper-complete-20261021':
  for d in (root/'runs').glob('*/*'):
   if not (d/'responses.jsonl').exists():files.extend(p for p in (d/'batches').glob('*.json') if not p.name.endswith(('.raw.json','.attempt.json')))
 with gzip.open(tmp,'wt',newline='',encoding='utf-8',compresslevel=3) as f:
  w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
  def emit(p,idx,r,raw):
   global n
   rel=p.relative_to(BASE);parts=rel.parts;j=r.get('judge','')
   if not j and 'judges' in parts:j=parts[parts.index('judges')+1].split('_')[0]
   if not j and 'nn-mechanism-two-labels' in name:j='gemma' if '-gemma-' in name else 'qwen' if '-qwen-' in name else 'qwen122'
   val=r.get('positive_probability','')
   if val=='' and isinstance(r.get('value'),(int,float)):val=r['value']
   w.writerow(dict(experiment=name,source_path=str(rel),source_row=idx,judge=j,backbone=r.get('backbone',''),dataset=r.get('dataset',''),split=r.get('split',''),sample_id=r.get('sample_id',r.get('unit_id','')),prompt_id=r.get('prompt_id',''),condition=r.get('condition',''),positive_probability=val,prediction=r.get('prediction',r.get('prediction_at_50','')),status=r.get('status',''),finish_reason=r.get('finish_reason',''),raw_response=r.get('raw_response',r.get('generated_text','')),raw_record_json=raw));n+=1
  for p in sorted(files):
   before=p.stat();h=hashlib.sha256();count=0
   if p.suffix=='.jsonl':
    with p.open('rb') as g:
     for line in g:
      h.update(line)
      if not line.strip():continue
      r=json.loads(line);count+=1;emit(p,count,r,line.decode().rstrip('\r\n'))
   else:
    data=p.read_bytes();h.update(data)
    for count,r in enumerate(json.loads(data)['records'],1):emit(p,count,r,json.dumps(r,ensure_ascii=False,separators=(',',':')))
   after=p.stat();stable=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
   if not stable:changes.append(str(p))
   sources.append({'path':str(p),'sha256':h.hexdigest(),'rows':count,'stable_during_read':stable})
 assert not changes,changes
 tmp.replace(dest);write(receipt,{'root':name,'file':dest.name,'bytes':dest.stat().st_size,'sha256':sha(dest),'rows':n,'sources':sources,'snapshot_time':time.time(),'active_snapshot':name=='paper-complete-20261021'});print(name,n,flush=True)
write(OUT/'COMPLETED.json',{'roots':ROOTS,'time':time.time(),'note':'Source-specific raw CSVs; counts include historical reuse across studies, not unique independent calls.'})
