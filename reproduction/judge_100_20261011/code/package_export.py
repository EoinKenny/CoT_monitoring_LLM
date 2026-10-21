from common import *
import zipfile,shutil
def main():
 d=ROOT/'api_export';m=read(d/'export_manifest.json')
 assert m['total']==60000 and sha(d/'prompts.csv')==m['prompts_csv_sha256']
 template=read(d/'model_manifest.template.json');template['prompts_csv_sha256']=m['prompts_csv_sha256'];write(d/'model_manifest.template.json',template)
 names=['README.txt','prompts.csv','responses_template.csv','model_manifest.template.json','validate_results.py','export_manifest.json']
 m['files']={n:sha(d/n) for n in names if n!='export_manifest.json'};write(d/'export_manifest.json',m)
 zpath=d/'external_api_judges_60000_20261011.zip';assert not zpath.exists()
 with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n in names:z.write(d/n,n)
 with zipfile.ZipFile(zpath) as z:
  assert z.namelist()==names and z.testzip() is None
  for n in names:
   h=hashlib.sha256()
   with z.open(n) as f:
    for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
   assert h.hexdigest()==sha(d/n)
 write(d/'PACKAGE_COMPLETED.json',dict(time=time.time(),rows=60000,archive=str(zpath),archive_bytes=zpath.stat().st_size,archive_sha256=sha(zpath),files=names))
 print(json.dumps(read(d/'PACKAGE_COMPLETED.json')),flush=True)
if __name__=='__main__':main()
