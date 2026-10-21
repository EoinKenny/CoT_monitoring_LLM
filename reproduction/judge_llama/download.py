import os,socket,json
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
old=socket.getaddrinfo
socket.getaddrinfo=lambda host,port,family=0,type=0,proto=0,flags=0:old(host,port,socket.AF_INET,type,proto,flags)
from huggingface_hub import HfApi,snapshot_download
root=Path('/home/ubuntu/experiments/judge-llama-20261002')
root.mkdir(exist_ok=True)
model='RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16'
revision='da23851b92f5ea8023d6c70f790feb8e05269812'
(root/'model_revision.json').write_text(json.dumps({'model':model,'revision':revision}))
print('Pinned',model,revision,flush=True)
p=snapshot_download(model,revision=revision,max_workers=4,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model'])
(root/'model_path.txt').write_text(p)
print('DOWNLOAD_COMPLETED',p,flush=True)
