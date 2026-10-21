from common import *
from transformers import AutoTokenizer
import numpy as np
OLD='Return exactly one digit: 1 for the positive class or 0 for the negative class. Do not include explanations or other text.'
FOOTERS=['Classification (0 or 1):','Classification of the CURRENT conversation (0 or 1):']
def main():
 reserve();assert not (ROOT/'PREPARED.json').exists();protocol=read(ROOT/'PROTOCOL.json');instruction=protocol['output_instruction']
 assert sha(PARENT/'RESULTS.json')==read(PARENT/'COMPLETED.json')['results_sha256']==read(PARENT/'EVALUATION_COMPLETED.json')['sha256']
 assert len(read(PARENT/'RESULTS.json'))==100
 audit={};sources={};total=0;smoke={j:{c:[] for c in CS} for j in JS};tokenizers={}
 def source(p):sources[str(p)]=sha(p)
 for b in BBS:
  P=PARENT/b;B=ROOT/b;B.mkdir(exist_ok=True);selected=read(P/'selected_flags.json');ids=[r['sample_id'] for r in selected]
  assert len(ids)==10000 and len(set(ids))==10000 and ids==read(P/'SAMPLE_FROZEN.json')['sample_ids'];assert sha(P/'selected_flags.json')==read(P/'SAMPLE_FROZEN.json')['selection_sha256']
  for rel in ['selected_flags.json','SAMPLE_FROZEN.json','THRESHOLDS.json']:
   source(P/rel);shutil.copy2(P/rel,B/rel)
  for ds in read(P/'THRESHOLDS.json'):
   p=P/f'{ds}_predictions.npz';source(p);(B/p.name).symlink_to(p);z=np.load(p)
   for r in selected:
    if r['dataset']==ds:
     i=r['index'];assert z['labels'][i]==r['true_label'] and z['splits'][i]==r['split'];assert float(z['probabilities'][i])==r['probe_probability'];assert 0<r['inclusion_probability']<=1
   source(P/f'{ds}_probe.joblib')
  for j in JS:
   for c in CS:
    D=P/'judges'/f'{j}_{c}';O=B/'judges'/f'{j}_{c}';O.mkdir(parents=True,exist_ok=True)
    parent_cfg=read(D/'run/config.json');parent_done=read(D/'run/COMPLETED');assert sha(D/'run/judgments.jsonl')==parent_done['judgments_sha256'];assert sha(D/'run/config.json')==parent_done['config_sha256']
    records=list(jsonlines(D/'run/judgments.jsonl'));assert [r['sample_id'] for r in records]==ids and all(r['prediction'] in [0,1] for r in records)
    for batch in sorted((D/'run/batches').glob('*.json')):
     if batch.name.endswith(('.attempt.json','.raw.json')):continue
     doc=read(batch);assert doc['raw_sha256']==sha(batch.with_suffix('.raw.json'));assert doc['config_sha256']==parent_done['config_sha256'];assert doc['records_sha256']==digest(json.dumps(doc['records'],sort_keys=True))
    prompts=list(jsonlines(D/'prompts.jsonl'));assert [r['sample_id'] for r in prompts]==ids;assert sha(D/'prompts.jsonl')==parent_cfg['prompt_sha256']
    path=(D/'model_path.txt').read_text().strip();tok=tokenizers.setdefault(j,AutoTokenizer.from_pretrained(path,local_files_only=True)) if j not in tokenizers else tokenizers[j]
    out=[]
    for p in prompts:
     text=p['prompt'];start=text.index(OLD);assert start<text.index('Classify this supplied conversation JSON:')
     footer=FOOTERS[c=='nearest'];end=text.rindex(footer);assert end>start
     new=text[:start]+instruction+text[start+len(OLD):end]+'Positive-class probability (0-100):'+text[end+len(footer):]
     # Reversibility proves that every other byte, including explanations and conversation, is preserved.
     restored=new.replace(instruction,OLD,1);k=restored.rindex('Positive-class probability (0-100):');restored=restored[:k]+footer+restored[k+len('Positive-class probability (0-100):'):];assert restored==text
     ti=tok.encode(new,add_special_tokens=False)
     out.append(dict(sample_id=p['sample_id'],dataset=p['dataset'],prompt=new,token_ids=ti,input_tokens=len(ti),instruction_sha256=digest(new),parent_instruction_sha256=p['instruction_sha256'],parent_prompt_sha256=digest(text),backbone=b,condition=c))
    lines(O/'prompts.jsonl',out);(O/'model_path.txt').write_text(path);source(D/'prompts.jsonl');source(D/'run/config.json');source(D/'run/judgments.jsonl');source(D/'run/COMPLETED')
    cfg=dict(model=parent_cfg['model'],revision=parent_cfg['revision'],judge=j,condition=c,backbone=b,n=len(out),prompt_sha256=sha(O/'prompts.jsonl'),parent_config_sha256=sha(D/'run/config.json'),parent_engine=parent_cfg['engine'],max_input_tokens=max(p['input_tokens'] for p in out),total_input_tokens=sum(p['input_tokens'] for p in out),protocol_sha256=sha(ROOT/'PROTOCOL.json'),output_instruction=instruction,tokenizer_path=path,tokenizer_files={p.name:sha(p) for p in Path(path).iterdir() if p.name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','config.json']})
    write(O/'approved_config.json',cfg);audit[f'{b}/{j}/{c}']=dict(n=len(out),prompt_sha256=cfg['prompt_sha256'],same_parent_cases_and_order=True,exact_noncontract_text_preserved=True,max_input_tokens=cfg['max_input_tokens']);total+=len(out)
    smoke[j][c].extend(out)
    print(json.dumps(dict(prepared=f'{b}/{j}/{c}',n=len(out))),flush=True)
 for j in JS:
  smoke_rows=[]
  for c in CS:
   pool=sorted(smoke[j][c],key=lambda r:(r['input_tokens'],r['backbone'],r['sample_id']));indices=sorted(set([round(i*(len(pool)-1)/31) for i in range(32)]));assert len(indices)==32
   for i in indices:
    r=dict(pool[i]);r['source_sample_id']=r['sample_id'];r['sample_id']=r['backbone']+':'+c+':'+r['sample_id'];smoke_rows.append(r)
  O=ROOT/'smoke'/j;O.mkdir(parents=True,exist_ok=True);lines(O/'prompts.jsonl',smoke_rows)
  D=ROOT/'llama/judges'/f'{j}_baseline';cfg=read(D/'approved_config.json');cfg.update(n=len(smoke_rows),prompt_sha256=sha(O/'prompts.jsonl'),max_input_tokens=max(p['input_tokens'] for p in smoke_rows),total_input_tokens=sum(p['input_tokens'] for p in smoke_rows),condition='smoke_all_three',backbone='smoke_all_five');write(O/'approved_config.json',cfg);shutil.copy2(D/'model_path.txt',O/'model_path.txt')
 assert total==300000;write(ROOT/'PARENT_VERIFIED.json',dict(time=time.time(),n=300000,sources=sources));write(ROOT/'PREPARED.json',dict(total=total,cells=audit,protocol_sha256=sha(ROOT/'PROTOCOL.json')))
if __name__=='__main__':main()
