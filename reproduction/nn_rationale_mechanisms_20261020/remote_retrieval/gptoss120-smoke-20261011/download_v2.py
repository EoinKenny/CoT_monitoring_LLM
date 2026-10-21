from pathlib import Path
import json,time,shutil,concurrent.futures
from huggingface_hub import hf_hub_download
R=Path('/home/ubuntu/experiments/gptoss120-smoke-20261011');repo='openai/gpt-oss-120b';rev='b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
assert shutil.disk_usage(R).free>100*1024**3
info=json.loads((R/'hf_metadata.json').read_text());assert info['sha']==rev
files=[x['rfilename'] for x in info['siblings'] if '/' not in x['rfilename'] and x['rfilename'].endswith(('.json','.jinja','.safetensors','.model','.txt'))]
(R/'download_manifest.json').write_text(json.dumps(dict(repo=repo,revision=rev,files=files,bytes=sum(x.get('size',0) for x in info['siblings'] if x['rfilename'] in files),repair='Initial HfApi metadata request stalled; use identical revision metadata fetched with bounded curl. No inference had started.'),indent=2))
t=time.time()
def get(f):
 p=hf_hub_download(repo,f,revision=rev);print('downloaded',f,flush=True);return p
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:paths=list(pool.map(get,files))
p=str(Path(paths[0]).parent);(R/'model_path.txt').write_text(p);(R/'DOWNLOAD_COMPLETED.json').write_text(json.dumps(dict(path=p,seconds=time.time()-t)));print('complete',p,flush=True)
