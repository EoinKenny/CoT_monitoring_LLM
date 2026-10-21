import sys,platform,subprocess,importlib.metadata
from common import *
import torch,transformers
name=sys.argv[1]
write(ROOT/(name+'_ENVIRONMENT.json'),dict(executable=sys.executable,python=sys.version,platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']),time=time.time()))
