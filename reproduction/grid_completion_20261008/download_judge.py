import pathlib,json,subprocess,os,hashlib,time,concurrent.futures,shutil,traceback
R=pathlib.Path('/home/ubuntu/experiments/grid-completion-20261008/qwen122_judge_download');REPO='Qwen/Qwen3.5-122B-A10B-GPTQ-Int4'
def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 R.mkdir(parents=True,exist_ok=True)
 subprocess.run(['curl','-fsSL','--connect-timeout','15','--max-time','60',f'https://huggingface.co/api/models/{REPO}?blobs=true','-o',str(R/'hub_info.json')],check=True)
 meta=json.loads((R/'hub_info.json').read_text());rev=meta['sha'];D=pathlib.Path('/opt/dlami/nvme/qwen122b-gptq-judge')/rev;D.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(D).free>100*2**30
 entries={s['rfilename']:s for s in meta['siblings'] if '/' not in s['rfilename'] and s['rfilename'].endswith(('.safetensors','.json','.jinja','.txt'))}
 names=set(entries);write('DOWNLOAD_PLAN.json',dict(repo=REPO,revision=rev,model_path=str(D),files=entries))
 def one(n):
  p=D/n;record=R/'downloads'/ (n+'.json');record.parent.mkdir(exist_ok=True)
  if record.exists():return json.loads(record.read_text())
  part=p.with_suffix(p.suffix+'.partial')
  subprocess.run(['curl','-fL','--connect-timeout','20','--max-time','3600','--retry','3','--retry-delay','3','--continue-at','-','--silent','--show-error',f'https://huggingface.co/{REPO}/resolve/{rev}/{n}','-o',str(part)],check=True)
  expected=entries[n].get('size');assert expected is None or part.stat().st_size==expected,(n,'size')
  h=sha(part);lfs=entries[n].get('lfs',{}).get('sha256');assert not lfs or h==lfs,(n,'sha256');os.replace(part,p);result=dict(name=n,bytes=p.stat().st_size,sha256=h,verified_lfs_sha256=bool(lfs));record.write_text(json.dumps(result));return result
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for f in concurrent.futures.as_completed([pool.submit(one,n) for n in sorted(names)]):
   results.append(f.result());write('DOWNLOAD_STATUS.json',dict(completed=len(results),total=len(names),bytes=sum(v['bytes'] for v in results),last=results[-1]['name'],time=time.time()));print(json.dumps(results[-1]),flush=True)
 write('DOWNLOAD_COMPLETED.json',dict(files=len(results),bytes=sum(v['bytes'] for v in results),time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:write('DOWNLOAD_FAILED_'+str(time.time_ns())+'.json',dict(error=str(e),traceback=traceback.format_exc()));raise
