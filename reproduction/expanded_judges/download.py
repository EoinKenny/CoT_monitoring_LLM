from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
import json,sys,time
R=Path('/home/ubuntu/experiments/expanded-judges-20261003');name=sys.argv[1]
repo={'glm':'zai-org/GLM-4.7-Flash','olmo':'allenai/Olmo-3.1-32B-Instruct'}[name]
rev=HfApi().model_info(repo).sha
(R/(name+'_download_request.json')).write_text(json.dumps(dict(repo=repo,revision=rev,started=time.time()),indent=2))
p=snapshot_download(repo_id=repo,revision=rev,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','*.tiktoken','README.md'],max_workers=4)
(R/(name+'_model.json')).write_text(json.dumps(dict(model=repo,revision=rev,path=p,quantization='vLLM 0.30 fp8_per_tensor online from official BF16 weights',enable_thinking=False),indent=2))
print(json.dumps(dict(model=repo,revision=rev,path=p)),flush=True)
