import sys,json,pathlib,subprocess,importlib.metadata
import torch,transformers
R=pathlib.Path('/home/ubuntu/project-migration-20261007');name=sys.argv[1]
assert torch.cuda.is_available()
x=torch.arange(1024,dtype=torch.float32,device='cuda');assert (x+x).cpu().tolist()==(2*torch.arange(1024)).tolist()
# Exercise matrix multiplication on Blackwell, not just device discovery.
a=torch.ones((64,64),dtype=torch.bfloat16,device='cuda');assert torch.all(a@a==64)
doc=dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
if name=='judge':
 import vllm
 doc['vllm']=vllm.__version__
else:
 import joblib,numpy as np
 b=pathlib.Path('/home/ubuntu/experiments/llama70b-full-20261005/llama70');f=joblib.load(b/'toxicchat_probe.joblib');x=np.load(b/'toxicchat_features.npy',mmap_mode='r')[:20];p=f['model'].predict_proba(f['scaler'].transform(x))[:,1];expected=np.load(b/'toxicchat_predictions.npz')['probabilities'][:20];assert np.allclose(p,expected,rtol=1e-10,atol=1e-10);doc['probe_replay_max_error']=float(np.max(abs(p-expected)))
(R/(name+'_RUNTIME_VERIFIED.json')).write_text(json.dumps(doc,indent=2));print(json.dumps(doc))
