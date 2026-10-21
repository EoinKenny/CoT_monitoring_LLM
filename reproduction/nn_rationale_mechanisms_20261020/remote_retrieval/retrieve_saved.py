"""Read-only remote archive stream. No remote files are created or modified."""
import argparse, subprocess, hashlib, json, tarfile, datetime
from pathlib import Path
R=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import sys,tarfile,hashlib,json,io,time
name=sys.argv[1];root=Path('/home/ubuntu/experiments')/name
files=[];excluded=[]
for p in sorted(root.rglob('*')):
 if not p.is_file(): continue
 rel=p.relative_to(root)
 if '__pycache__' in rel.parts or p.suffix in ['.npz','.lock'] or p.name in ['prompts.jsonl','cases.jsonl','selected_flags.json','THRESHOLDS.json','SAMPLE_FROZEN.json']:
  excluded.append({'path':str(rel),'bytes':p.stat().st_size});continue
 files.append(p)
manifest={'root':str(root),'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'read_only':True,'files':[],'excluded':excluded,'exclusions_note':'Prompt/case bodies and frozen numeric arrays excluded from this first structured-output transfer; exact output raw and parsed and attempt records retained. No source was modified.'}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for p in files:
  b=p.read_bytes();rel=str(p.relative_to(root));manifest['files'].append({'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
  info=tarfile.TarInfo(name+'/'+rel);info.size=len(b);info.mtime=int(p.stat().st_mtime);tar.addfile(info,io.BytesIO(b))
 b=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo(name+'/RETRIEVAL_MANIFEST.json');info.size=len(b);tar.addfile(info,io.BytesIO(b))
'''
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
archive=R/(args.run+'.tar.gz');assert not archive.exists(),archive
cmd=['ssh','-i','/Users/eoinkenny/.ssh/llm_interpretable.pem','-o','BatchMode=yes','-o','ConnectTimeout=12','ubuntu@ec2-52-14-51-0.us-east-2.compute.amazonaws.com','python3 - '+args.run]
with archive.open('xb') as out:
 proc=subprocess.run(cmd,input=REMOTE.encode(),stdout=out,stderr=subprocess.PIPE,timeout=600)
if proc.returncode: raise RuntimeError(proc.stderr.decode())
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  assert m.name.startswith(args.run+'/') and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 tar.extractall(R,filter='data')
folder=R/args.run;manifest=json.loads((folder/'RETRIEVAL_MANIFEST.json').read_text())
for item in manifest['files']:
 b=(folder/item['path']).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],item['path']
receipt={'run':args.run,'file_count':len(manifest['files']),'total_bytes':sum(x['bytes'] for x in manifest['files']),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'all_downloaded_file_hashes_verified':True,'source_read_only':True,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/(args.run+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
