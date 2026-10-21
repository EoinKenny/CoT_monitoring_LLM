"""Verify local data archives against their original transfer receipts."""
from pathlib import Path
import argparse,json,hashlib
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def verify(root,analysis_only=False):
 local=json.loads((root/'LOCAL_MANIFEST.json').read_text());p=root/local['archive'];assert p.stat().st_size==local['bytes'];assert sha(p)==local['sha256'],p
 archives=[p];print('Verified local analysis archive',flush=True)
 if not analysis_only:
  for r in sorted((root/'remote_archives').glob('*.DONE.json')):
   d=json.loads(r.read_text())
   for key in ['archive','csv']:
    p=r.parent/d[key];assert p.stat().st_size==d[key+'_bytes'];assert sha(p)==d[key+'_sha256'],p
   archives.append(r.parent/d['archive']);print('Verified',d['root'],flush=True)
  complete=root/'remote_archives/COMPLETED.json'
  if complete.exists():assert len(archives)-1==len(json.loads(complete.read_text())['roots'])
  else:print('Snapshot transfer is partial; remote COMPLETED.json is absent.')
 return archives
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backup',type=Path,required=True);p.add_argument('--analysis-only',action='store_true');a=p.parse_args();verify(a.backup,a.analysis_only)
