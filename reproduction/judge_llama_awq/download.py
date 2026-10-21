import os,socket,json
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
old=socket.getaddrinfo
socket.getaddrinfo=lambda host,port,family=0,type=0,proto=0,flags=0:old(host,port,socket.AF_INET,type,proto,flags)
from huggingface_hub import HfApi,snapshot_download
root=Path('/home/ubuntu/experiments/judge-llama-awq-20261002')
root.mkdir(exist_ok=True)
model='ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4'
revision='a3c28ea93c041acfaadb85401ad6493749a5aa04'
(root/'model_revision.json').write_text(json.dumps({'model':model,'revision':revision}))
print('Pinned',model,revision,flush=True)
p=snapshot_download(model,revision=revision,max_workers=4,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model'])
(root/'model_path.txt').write_text(p)
print('DOWNLOAD_COMPLETED',p,flush=True)
