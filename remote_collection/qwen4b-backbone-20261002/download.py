from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
import json,time
R=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002');R.mkdir(exist_ok=True)
assert not (R/'backbone.json').exists()
repo='Qwen/Qwen3-4B-Instruct-2507';rev=HfApi().model_info(repo).sha
p=snapshot_download(repo_id=repo,revision=rev,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','README.md'])
(R/'backbone.json').write_text(json.dumps(dict(repo=repo,revision=rev,path=p,block_index=24,relative_depth='25/36 blocks, matching Llama 11/16 to nearest block',dtype='bfloat16'),indent=2))
print((R/'backbone.json').read_text())
