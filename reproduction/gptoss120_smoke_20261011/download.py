from pathlib import Path
import json,time,shutil
from huggingface_hub import snapshot_download,HfApi
R=Path('/home/ubuntu/experiments/gptoss120-smoke-20261011');R.mkdir(exist_ok=True)
repo='openai/gpt-oss-120b';rev='b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
assert shutil.disk_usage(R).free>100*1024**3
info=HfApi().model_info(repo,revision=rev,files_metadata=True)
files=[x.rfilename for x in info.siblings if '/' not in x.rfilename and (x.rfilename.endswith(('.json','.jinja','.safetensors','.model','.txt')))]
(R/'download_manifest.json').write_text(json.dumps(dict(repo=repo,revision=rev,files=files,bytes=sum(x.size or 0 for x in info.siblings if x.rfilename in files)),indent=2))
t=time.time();p=snapshot_download(repo,revision=rev,allow_patterns=files,max_workers=8)
(R/'model_path.txt').write_text(p)
(R/'DOWNLOAD_COMPLETED.json').write_text(json.dumps(dict(path=p,seconds=time.time()-t)))
print(p,flush=True)
