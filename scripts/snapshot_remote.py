"""Non-destructive experiment snapshot and raw CSV export. Never runs inference."""
from pathlib import Path
import os,sys,json,hashlib,tarfile,gzip,csv,time,shutil,io,traceback
BASE=Path(os.environ.get('EXPERIMENTS_ROOT','/home/ubuntu/experiments'))
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
ROOTS=json.loads(Path(sys.argv[2]).read_text())
SKIP_DIR={'.git','.venv','.venv-uv','.venv-vllm','__pycache__','node_modules','site-packages','model-cache','models--','api_export'}
SKIP_EXT={'.safetensors','.bin','.pt','.pth','.sqlite','.zip','.tar','.tgz'}
CSV_NAMES={'judgments.jsonl','responses.jsonl','annotations.jsonl','reused.jsonl'}
FIELDS=['experiment','source_path','source_row','judge','backbone','dataset','split','sample_id','prompt_id','condition','positive_probability','prediction','status','finish_reason','raw_response','raw_record_json']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def reserve():assert shutil.disk_usage(OUT).free>25*2**30,'Disk reserve reached'
class Reader:
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.n+=len(b);return b
def judge_for(rel,r):
 if r.get('judge'):return r['judge']
 parts=rel.parts
 if 'judges' in parts:return parts[parts.index('judges')+1].split('_')[0]
 name=parts[0]
 if name=='nn-mechanism-two-labels-20261018':return 'qwen122'
 for j in ['qwen','gemma']:
  if name=='nn-mechanism-two-labels-'+j+'-20261018':return j
 return ''
def main():
 for ix,name in enumerate(ROOTS):
  reserve();root=BASE/name
  if not root.exists():write(OUT/(name+'.MISSING.json'),{'root':str(root)});continue
  receipt=OUT/(name+'.DONE.json')
  if receipt.exists():continue
  write(OUT/'STATUS.json',{'state':'snapshotting','index':ix,'root':name,'time':time.time()})
  archive=OUT/(name+'.tar.gz');tmp=archive.with_suffix('.gz.partial');cp=OUT/(name+'.raw.csv.gz');ct=cp.with_suffix('.gz.partial')
  assert not tmp.exists() and not archive.exists() and not ct.exists()
  manifest=[];excluded=[];links=[];csv_counts={};changes=[]
  with tarfile.open(tmp,'w:gz',compresslevel=3,dereference=False) as tar,gzip.open(ct,'wt',encoding='utf-8',newline='',compresslevel=3) as cf:
   writer=csv.DictWriter(cf,fieldnames=FIELDS);writer.writeheader()
   def rows_to_csv(rel,rows):
    count=0
    for lineno,r in enumerate(rows,1):
     if not isinstance(r,dict):continue
     prob=r.get('positive_probability','')
     if prob=='' and isinstance(r.get('value'),(int,float)):prob=r['value']
     writer.writerow(dict(experiment=name,source_path=str(rel),source_row=lineno,judge=judge_for(rel,r),backbone=r.get('backbone',''),dataset=r.get('dataset',''),split=r.get('split',''),sample_id=r.get('sample_id',r.get('unit_id','')),prompt_id=r.get('prompt_id',''),condition=r.get('condition',''),positive_probability=prob,prediction=r.get('prediction',r.get('prediction_at_50','')),status=r.get('status',''),finish_reason=r.get('finish_reason',''),raw_response=r.get('raw_response',r.get('generated_text','')),raw_record_json=json.dumps(r,ensure_ascii=False,separators=(',',':'))));count+=1
    csv_counts[str(rel)]=count
   for directory,dirs,files in os.walk(root,followlinks=False):
    for dirname in dirs:
     dp=Path(directory)/dirname
     if dp.is_symlink():
      rel=dp.relative_to(BASE);tar.add(dp,arcname=str(rel),recursive=False);links.append({'path':str(rel),'target':os.readlink(dp),'resolved':str(dp.resolve()),'directory':True})
    dirs[:]=sorted(d for d in dirs if d not in SKIP_DIR and not d.startswith('.venv'))
    for filename in sorted(files):
     p=Path(directory)/filename;rel=p.relative_to(BASE)
     if p.suffix in SKIP_EXT or filename.endswith(('.tmp','.partial','.lock','.pyc')):
      excluded.append({'path':str(rel),'reason':'weights, packaged duplicate, transient or lock'});continue
     if p.suffix=='.npy' and p.stat().st_size>100_000_000:
      excluded.append({'path':str(rel),'bytes':p.stat().st_size,'reason':'large activation/cache array; regeneration required'});continue
     if p.is_symlink():
      tar.add(p,arcname=str(rel),recursive=False);links.append({'path':str(rel),'target':os.readlink(p),'resolved':str(p.resolve())});continue
     try:
      before=p.stat();info=tar.gettarinfo(str(p),arcname=str(rel))
      with p.open('rb') as source:reader=Reader(source);tar.addfile(info,reader)
      after=p.stat();entry={'path':str(rel),'bytes':reader.n,'sha256':reader.h.hexdigest()};manifest.append(entry)
      changed=(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)
      if changed:changes.append(str(rel));continue
      if p.name in CSV_NAMES:
       with p.open() as f:rows_to_csv(rel,(json.loads(line) for line in f if line.strip()))
      elif name=='paper-complete-20261021' and p.parent.name=='batches' and p.suffix=='.json' and not p.name.endswith(('.raw.json','.attempt.json')) and not (p.parent.parent/'responses.jsonl').exists():
       rows_to_csv(rel,json.loads(p.read_text())['records'])
     except FileNotFoundError:changes.append(str(rel)+' disappeared during live snapshot')
    reserve()
   data=json.dumps({'files':manifest,'symlinks':links,'excluded':excluded,'changed_during_snapshot':changes,'csv_rows_by_source':csv_counts,'time':time.time()},indent=2).encode();info=tarfile.TarInfo(name+'/BACKUP_MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
  tmp.replace(archive);ct.replace(cp)
  write(receipt,{'root':name,'time':time.time(),'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'csv':cp.name,'csv_bytes':cp.stat().st_size,'csv_sha256':sha(cp),'files':len(manifest),'uncompressed_bytes':sum(r['bytes'] for r in manifest),'csv_rows':sum(csv_counts.values()),'excluded':excluded,'symlinks':links,'changed_during_snapshot':changes,'active_snapshot':name=='paper-complete-20261021'})
  print(name,'DONE',archive.stat().st_size,'CSV rows',sum(csv_counts.values()),flush=True)
 write(OUT/'COMPLETED.json',{'time':time.time(),'roots':ROOTS,'scope':'point-in-time; active collection needs a final incremental snapshot'})
if __name__=='__main__':
 try:main()
 except BaseException:write(OUT/'FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
