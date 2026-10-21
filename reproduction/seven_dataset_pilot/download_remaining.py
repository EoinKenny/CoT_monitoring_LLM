import os,socket,json,urllib.request,time
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
old=socket.getaddrinfo;socket.getaddrinfo=lambda h,p,family=0,type=0,proto=0,flags=0:old(h,p,socket.AF_INET,type,proto,flags)
from huggingface_hub import HfApi,hf_hub_download
R=Path('/home/ubuntu/experiments/seven-dataset-pilot-20261002');R.mkdir(exist_ok=True);api=HfApi();manifest=json.loads((R/'downloads.json').read_text()) if (R/'downloads.json').exists() else {}
repos={'beavertails':('PKU-Alignment/BeaverTails',None),'wildjailbreak':('allenai/wildjailbreak',['train/train.tsv','eval/eval.tsv']),'aegis':('nvidia/Aegis-AI-Content-Safety-Dataset-2.0',['train.json','validation.json','test.json']),'toxigen':('toxigen/toxigen-data',['annotated/train-00000-of-00001.parquet','annotated/test-00000-of-00001.parquet']),'fava':('fava-uw/fava-data',['training.json','annotations.json'])}
for name,(repo,files) in repos.items():
 if name=='wildjailbreak':continue # Await legitimate account access; original denial preserved.
 if name in manifest and manifest[name]['files']:continue
 info=api.dataset_info(repo);rev=info.sha
 if files is None:files=[s.rfilename for s in info.siblings if '30k' in s.rfilename and '330k' not in s.rfilename]
 manifest[name]={'repo':repo,'revision':rev,'files':{}}
 for f in files:
  if not f.endswith(('.json','.jsonl','.jsonl.gz','.parquet','.tsv','.json.gz')):continue
  print('download',name,f,flush=True);manifest[name]['files'][f]=hf_hub_download(repo,f,repo_type='dataset',revision=rev)
 (R/'downloads.json').write_text(json.dumps(manifest,indent=2))
for name,repo,files in [('halueval','RUCAIBox/HaluEval',['data/qa_data.json']),('ragtruth','ParticleMedia/RAGTruth',['dataset/response.jsonl','dataset/source_info.jsonl'])]:
 req=urllib.request.Request('https://api.github.com/repos/'+repo+'/commits/main',headers={'User-Agent':'research-pilot'})
 rev=json.load(urllib.request.urlopen(req))['sha'];manifest[name]={'repo':repo,'revision':rev,'files':{}}
 for f in files:
  p=R/(name+'_'+f.split('/')[-1]);print('download',name,f,flush=True);urllib.request.urlretrieve('https://raw.githubusercontent.com/'+repo+'/'+rev+'/'+f,p);manifest[name]['files'][f]=str(p)
 (R/'downloads.json').write_text(json.dumps(manifest,indent=2))
(R/'DOWNLOAD_COMPLETED').write_text('ok')
