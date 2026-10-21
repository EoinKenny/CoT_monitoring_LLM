from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import json,time,socket
# Avoid long connection stalls on unreachable IPv6 routes on this instance.
original_getaddrinfo=socket.getaddrinfo
def ipv4_getaddrinfo(host,port,family=0,type=0,proto=0,flags=0):
 return original_getaddrinfo(host,port,socket.AF_INET if family in (0,socket.AF_UNSPEC) else family,type,proto,flags)
socket.getaddrinfo=ipv4_getaddrinfo
R=Path('/home/ubuntu/experiments/gemma31b-backbone-20261003')
repo='google/gemma-4-31B-it'
rev=HfApi().model_info(repo,timeout=60).sha
(R/'download_request.json').write_text(json.dumps(dict(repo=repo,revision=rev,started=time.time()),indent=2))
index=json.load(open(hf_hub_download(repo,'model.safetensors.index.json',revision=rev)))
prefix='model.language_model.'
def needed(k):
 if not k.startswith(prefix):return False
 key=k[len(prefix):]
 return key=='embed_tokens.weight' or (key.startswith('layers.') and int(key.split('.')[1])<=40)
keys={k:v for k,v in index['weight_map'].items() if needed(k)}
assert any('layers.40.' in k for k in keys) and any('embed_tokens.weight' in k for k in keys)
files=sorted(set(keys.values()))
p=snapshot_download(repo,revision=rev,allow_patterns=files+['*.json','*.jinja','*.model','*.txt','README.md'],max_workers=4)
(R/'backbone.json').write_text(json.dumps(dict(model=repo,revision=rev,path=p,block_index=40,hidden_size=5376,dtype='bfloat16',quantization=None,loaded_shards=files,weight_keys=keys,extraction='Exact first 41 decoder blocks; unchanged original 60-layer configuration; CPU embedding lookup then BF16 transfer; later layers and output head not needed.'),indent=2))
print('Pinned BF16 backbone prefix ready',rev,flush=True)
