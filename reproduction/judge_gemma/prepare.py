from pathlib import Path
import json,hashlib
from transformers import AutoTokenizer
R=Path('/home/ubuntu/experiments/judge-gemma-20261002');Q=Path('/home/ubuntu/experiments/judge-full-20261001');B=Path('/home/ubuntu/experiments/logistic-probes-20261001')
assert not (R/'prompts.jsonl').exists()
config=json.loads((Q/'approved_config.json').read_text());config.update(json.loads((R/'model_revision.json').read_text()));config.update(reused=0,comparison='Same instructions, examples and hard-label rule as Qwen; native Gemma chat template and tokenizer')
tok=AutoTokenizer.from_pretrained(str(Path('/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots')/config['revision']),local_files_only=True)
assert tok.chat_template and 'enable_thinking' in tok.chat_template
rows=json.loads((Q/'private.json').read_text());metadata={name:json.loads((B/f'{name}_metadata.json').read_text()) for name in ['wildguard','high_stakes','toxicchat']};prompts=[]
for r in rows:
 name=r['dataset'];m=metadata[name][r['index']]
 messages=[dict(role='system',content=config['common']+'\n\n'+config['rubrics'][name]),dict(role='user',content='Classify this supplied conversation JSON:\n'+json.dumps(m['messages'],ensure_ascii=False)+'\n\nClassification (0 or 1):')]
 text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 ids=tok.encode(text,add_special_tokens=False)
 prompts.append(dict(sample_id=r['sample_id'],dataset=name,prompt=text,input_tokens=len(ids),token_ids=ids,instruction_sha256=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()))
assert len(prompts)==5190
(R/'prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in prompts));config['prompt_sha256']=hashlib.sha256((R/'prompts.jsonl').read_bytes()).hexdigest();config['label_ids']=[tok.encode(s,add_special_tokens=False) for s in ['0','1']]
assert all(len(x)==1 for x in config['label_ids'])
(R/'approved_config.json').write_text(json.dumps(config,indent=2));(R/'private.json').write_bytes((Q/'private.json').read_bytes());(R/'weighted_results.json').write_bytes((Q/'weighted_results.json').read_bytes())
print(json.dumps(dict(total=len(prompts),max_tokens=max(p['input_tokens'] for p in prompts),label_ids=config['label_ids'],template_suffix=prompts[0]['prompt'][-100:])))
