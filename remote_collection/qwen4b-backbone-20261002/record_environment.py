from pathlib import Path
import importlib.metadata,json,hashlib,sys,platform,torch,subprocess
R=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002')
info=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),nvidia_smi=subprocess.check_output(['nvidia-smi','-q'],text=True))
(R/'provenance/judge_environment.json').write_text(json.dumps(info,indent=2))
files=list(R.glob('*.py'))+[R/'README.md',R/'SAMPLE_FROZEN.json',R/'private.json']
(R/'provenance/sampled_code_checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print('Environment and source checksums recorded.')
