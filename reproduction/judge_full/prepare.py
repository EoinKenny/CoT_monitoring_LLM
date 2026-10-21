from pathlib import Path
import json,hashlib
import numpy as np
from transformers import AutoTokenizer
P=Path('/home/ubuntu/experiments/judge-pilot-20261001');R=Path('/home/ubuntu/experiments/judge-full-20261001');R.mkdir(exist_ok=True)
assert not (R/'prompts.jsonl').exists()
B=Path('/home/ubuntu/experiments/logistic-probes-20261001');W=Path('/home/ubuntu/experiments/logistic-weighted-20261001')
cfg=json.loads((P/'qwen36_v1/config.json').read_text());tok=AutoTokenizer.from_pretrained((P/'model_path.txt').read_text().strip(),local_files_only=True)
private=json.loads((P/'sample_private.json').read_text())['records'];pj={r['sample_id']:r for r in map(json.loads,(P/'qwen36_vllm_v3/judgments.jsonl').read_text().splitlines())};pp={r['sample_id']:r for r in map(json.loads,(P/'qwen36_v1/prompts.jsonl').read_text().splitlines())};pt=json.loads((P/'approved_token_ids.json').read_text());pm={(r['dataset'],r['index']):r['sample_id'] for r in private}
rows=[];prompts=[];reused=[];hashes={}
for name in ['wildguard','high_stakes','toxicchat']:
 path=W/f'{name}_test_predictions.npz';hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();saved=np.load(path);meta=json.loads((B/f'{name}_metadata.json').read_text())
 for j in np.flatnonzero(saved['predictions']==1):
  ix=int(saved['indices'][j]);m=meta[ix];sid=f'{name}-{ix}'
  rows.append(dict(sample_id=sid,dataset=name,index=ix,id=m['id'],source=m['source'],true_label=int(saved['labels'][j]),probe_prediction=1,probe_probability=float(saved['probabilities'][j])))
  messages=[dict(role='system',content=cfg['common']+'\n\n'+cfg['rubrics'][name]),dict(role='user',content='Classify this supplied conversation JSON:\n'+json.dumps(m['messages'],ensure_ascii=False)+'\n\nClassification (0 or 1):')]
  text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(text,add_special_tokens=False)
  prompts.append(dict(sample_id=sid,dataset=name,prompt=text,input_tokens=len(ids),token_ids=ids))
  if (name,ix) in pm:
   old=pm[name,ix];assert text==pp[old]['prompt'] and ids==pt[old]
   r=dict(pj[old]);r.update(sample_id=sid,reused_from=old);reused.append(r)
assert len(rows)==5190 and len(reused)==300
(R/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompts));(R/'reused.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reused));(R/'private.json').write_text(json.dumps(rows))
cfg.update(prediction_sha256=hashes,scope='All flagged test sequences; unflagged predictions unchanged',total=5190,reused=300,prompt_sha256=hashlib.sha256((R/'prompts.jsonl').read_bytes()).hexdigest())
(R/'approved_config.json').write_text(json.dumps(cfg,indent=2));(R/'weighted_results.json').write_bytes((W/'results.json').read_bytes());(R/'model_path.txt').write_bytes((P/'model_path.txt').read_bytes())
print(json.dumps(dict(total=len(rows),reused=len(reused),new=len(rows)-len(reused),max_tokens=max(r['input_tokens'] for r in prompts))))
