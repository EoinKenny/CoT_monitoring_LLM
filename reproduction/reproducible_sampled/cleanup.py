"""Delete only approved redundant weight objects and pip cache files; preserve metadata."""
import os,subprocess,signal
from pathlib import Path
from common import *
ROOT.mkdir(exist_ok=True)
old=E/'hard-negative-repro-20260928T082816Z/hf-cache/hub';hub=Path('/home/ubuntu/.cache/huggingface/hub')
targets=[old/'models--Qwen--Qwen3-30B-A3B',old/'models--zai-org--GLM-4.7-Flash',hub/'models--RedHatAI--Llama-3.3-70B-Instruct-quantized.w4a16']
assert not (ROOT/'CLEANUP_COMPLETED.json').exists()
objects={};links=[]
for target in targets:
 assert target.is_dir(),target
 for p in target.glob('snapshots/*/*'):
  if p.suffix not in ['.safetensors','.bin','.pt','.pth']:continue
  q=p.resolve(strict=True);assert q.is_relative_to(old) or q.is_relative_to(hub)
  st=q.stat();assert st.st_nlink==1,(q,'hard links require review')
  objects[str(q)]=dict(path=str(q),bytes=st.st_size,inode=st.st_ino,device=st.st_dev,mtime_ns=st.st_mtime_ns)
  if p.is_symlink():links.append(str(p))
# Ensure no retained snapshot references a deleted blob (including cross-cache links).
for cache in [old,hub]:
 for p in cache.glob('models--*/snapshots/*/*'):
  if p.is_symlink() and str(p.resolve()) in objects:assert any(p.is_relative_to(t) for t in targets),(p,'retained reference')
# Ensure no process has a target file open or mapped.
inodes={(v['device'],v['inode']) for v in objects.values()}
for proc in Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():
   try:s=fd.stat();assert (s.st_dev,s.st_ino) not in inodes,(proc,fd,'active weight')
   except (FileNotFoundError,PermissionError):pass
  maps=(proc/'maps').read_text()
  assert not any(p in maps for p in objects),(proc,'mapped weight')
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
pip=Path('/home/ubuntu/.cache/pip');pipfiles=[]
for p in pip.rglob('*'):
 if p.is_file() and not p.is_symlink():
  s=p.stat();assert s.st_nlink==1;pipfiles.append(dict(path=str(p),bytes=s.st_size,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns))
manifest=dict(time=time.time(),weight_objects=list(objects.values()),weight_links=links,pip_files=pipfiles,bytes=sum(v['bytes'] for v in objects.values())+sum(v['bytes'] for v in pipfiles))
write(ROOT/'CLEANUP_MANIFEST.json',manifest)
assert 150*2**30<manifest['bytes']<165*2**30,manifest['bytes']
for item in list(objects.values())+pipfiles:
 p=Path(item['path']);s=p.stat();assert (s.st_dev,s.st_ino,s.st_mtime_ns)==(item['device'],item['inode'],item['mtime_ns']);p.unlink()
for p in links:Path(p).unlink()
write(ROOT/'CLEANUP_COMPLETED.json',dict(bytes_removed=manifest['bytes'],free_bytes=shutil.disk_usage(ROOT).free,manifest_sha256=sha(ROOT/'CLEANUP_MANIFEST.json')))
# Retire only the already-held obsolete driver. Never touch keepalives.
pid=393599;proc=Path('/proc')/str(pid)
if proc.exists():
 cmd=(proc/'cmdline').read_bytes();assert b'/gemma31b-backbone-20261003/judge_stage.py' in cmd
 assert '\nState:\tT' in (proc/'status').read_text()
 assert not (proc/'task'/str(pid)/'children').read_text().strip()
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
write(ROOT/'OLD_CONTROLLER_RETIRED.json',dict(pid=pid,time=time.time()))
print(json.dumps(read(ROOT/'CLEANUP_COMPLETED.json')))
