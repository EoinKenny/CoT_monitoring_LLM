"""Verify local data archives against their original transfer receipts."""
from pathlib import Path
import argparse,json,hashlib,gzip,csv
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
  dep=root/'DEPENDENCIES_VERIFIED.json'
  if dep.exists():
   d=json.loads(dep.read_text());p=root/d['archive'];assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'];archives.append(p);print('Verified external artifact dependencies',flush=True)
 return archives
def verify_csv(root):
 directory=root/'paper_ready_csv';complete=json.loads((directory/'COMPLETED.json').read_text());rows=0;files=[]
 for name in complete['roots']:
  d=json.loads((directory/(name+'.VERIFIED.json')).read_text());p=directory/d['file'];assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256'],p
  with gzip.open(p,'rt',newline='') as f:
   reader=csv.DictReader(f);assert 'raw_record_json' in reader.fieldnames;n=sum(1 for _ in reader)
  assert n==d['rows'],p;rows+=n;files.append({'file':p.name,'rows':n,'sha256':d['sha256']})
 for d in json.loads((directory/'MANIFEST.json').read_text())['files']:assert sha(directory/d['file'])==d['sha256']
 result={'roots':len(files),'raw_rows_across_historical_studies':rows,'files':files,'note':'Historical reuse is retained; this row total is not a count of independent calls. Active collection is a snapshot.'}
 (root/'PAPER_CSV_VERIFIED.json').write_text(json.dumps(result,indent=2));print('Verified',len(files),'raw CSVs and original API/frozen-case exports;',rows,'historical rows')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backup',type=Path,required=True);p.add_argument('--analysis-only',action='store_true');p.add_argument('--csv-only',action='store_true');a=p.parse_args();verify_csv(a.backup) if a.csv_only else verify(a.backup,a.analysis_only)
