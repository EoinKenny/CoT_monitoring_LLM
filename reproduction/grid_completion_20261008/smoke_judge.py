"""Independent full-model pilot: longest and shortest prompts from each condition."""
from common import *
import subprocess,sys
directory=ROOT/'smoke/qwen122_pilot';directory.mkdir(parents=True,exist_ok=True)
assert not (directory/'run').exists(),'A previous pilot exists; review instead of silent regeneration'
chosen=[]
for c in CONDITIONS:
 pool=[]
 for b in OLD:
  path=ROOT/b/'judges'/f'qwen122_{c}'/'prompts.jsonl'
  for line in path.open():
   r=json.loads(line);r['sample_id']=b+':'+c+':'+r['sample_id'];pool.append(r)
 for r in [min(pool,key=lambda r:r['input_tokens']),max(pool,key=lambda r:r['input_tokens'])]:
  if r['sample_id'] not in {v['sample_id'] for v in chosen}:chosen.append(r)
with (directory/'prompts.jsonl').open('x') as f:
 for r in chosen:f.write(json.dumps(r)+'\n')
source=B/'judges/qwen122_baseline';cfg=read(source/'approved_config.json');cfg.update(n=len(chosen),prompt_sha256=sha(directory/'prompts.jsonl'),purpose='Independent engine smoke; outputs are not inserted into canonical results')
write(directory/'approved_config.json',cfg);(directory/'model_path.txt').write_bytes((source/'model_path.txt').read_bytes())
subprocess.run([sys.executable,str(ROOT/'code/judge.py'),str(directory)],check=True)
write(ROOT/'QWEN122_JUDGE_SMOKE_PASSED.json',dict(n=len(chosen),max_input_tokens=max(r['input_tokens'] for r in chosen),outputs_sha256=sha(directory/'run/judgments.jsonl'),quantization='gptq_marlin',time=time.time()))
