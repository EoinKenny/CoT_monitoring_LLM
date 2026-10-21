"""Restore verified data to an empty workspace without following archive symlinks."""
from pathlib import Path,PurePosixPath
import argparse,tarfile,json,shutil,os,hashlib,posixpath
from verify_backup import verify
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive member: '+name)
 target=root.joinpath(*p.parts)
 assert target.resolve().is_relative_to(root.resolve())
 return target
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backup',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--analysis-only',action='store_true');a=p.parse_args()
 assert not a.out.exists(),'Restore to a new directory';archives=verify(a.backup,a.analysis_only);a.out.mkdir(parents=True);skipped=[]
 # Avoid a full decompression pass just to discover required space. Check while extracting.
 for index,archive in enumerate(archives):
  root=a.out if index==0 or archive.name=='external_dependencies.tar.gz' else a.out/'remote_experiments';root.mkdir(exist_ok=True)
  with tarfile.open(archive,'r|gz') as t:
   for member in t:
    dest=safe(root,member.name)
    if member.isdir():dest.mkdir(parents=True,exist_ok=True)
    elif member.isfile():
     assert shutil.disk_usage(a.out).free>member.size+20*2**30,'Insufficient space: preserve 20 GiB free'
     dest.parent.mkdir(parents=True,exist_ok=True)
     assert not dest.exists(),'Duplicate archive file: '+str(dest)
     with t.extractfile(member) as source,dest.open('xb') as f:shutil.copyfileobj(source,f,2**20)
    elif member.issym() or member.islnk():skipped.append({'archive':archive.name,'path':member.name,'target':member.linkname})
  print('Restored',archive.name,flush=True)
 # Resolve only links whose targets are also inside the restored experiment tree.
 unresolved=[];remote=a.out/'remote_experiments'
 mapping=a.out/'EXTERNAL_DEPENDENCY_MAPPING.json';external={r['source']:safe(a.out,r['path']) for r in json.loads(mapping.read_text())} if mapping.exists() else {}
 for link in skipped:
  name=link['path'];target=link['target']
  if link['archive']==archives[0].name:unresolved.append(link);continue
  dest=safe(remote,name)
  original_target=target if target.startswith('/') else posixpath.normpath('/home/ubuntu/experiments/'+posixpath.dirname(name)+'/'+target)
  if original_target in external:candidate=external[original_target]
  elif target.startswith('/home/ubuntu/experiments/'):candidate=safe(remote,target.removeprefix('/home/ubuntu/experiments/'))
  elif not target.startswith('/'):candidate=(dest.parent/target).resolve()
  else:unresolved.append(link);continue
  if candidate.resolve().is_relative_to(a.out.resolve()) and candidate.exists() and not dest.exists():
   dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(os.path.relpath(candidate,dest.parent),target_is_directory=candidate.is_dir())
  else:unresolved.append(link)
 (a.out/'RESTORE_REPORT.json').write_text(json.dumps({'archives':[x.name for x in archives],'unresolved_links':unresolved,'note':'Large model weights and activation arrays may be intentionally absent; see each backup manifest.'},indent=2))
 print('Unresolved links:',len(unresolved))
if __name__=='__main__':main()
