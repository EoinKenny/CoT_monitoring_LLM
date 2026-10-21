from judge import ROOT,SOURCE,BBS,CS,sha,save,lines
from pathlib import Path
import json,hashlib,collections,random
from transformers import AutoTokenizer
PARENT=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009')
SEED=20261011
def rank(s):return hashlib.sha256((str(SEED)+'|full-pilot|'+s).encode()).hexdigest()
def main():
 assert not (ROOT/'PREPARED.json').exists()
 protocol=json.loads((ROOT/'PROTOCOL.json').read_text());n=protocol['per_class_per_dataset']
 source_hashes={};selected=[];selection_audit=[]
 def track(p):source_hashes[str(p)]=sha(p)
 pools={};thresholds={}
 for b in BBS:
  p=SOURCE/b/'selected_flags.json';track(p);pools[b]=json.loads(p.read_text())
  p=PARENT/b/'THRESHOLDS.json';track(p);thresholds[b]=json.loads(p.read_text())
 datasets=sorted({r['dataset'] for r in pools[BBS[0]]})
 for di,ds in enumerate(datasets):
  for label in [0,1]:
   # Near-equal backbone allocation with rotating remainder; class balance is exact per dataset.
   quotas={b:n//len(BBS) for b in BBS}
   for k in range(n%len(BBS)):quotas[BBS[(di+label+k)%len(BBS)]]+=1
   for b in BBS:
    eligible=[r for r in pools[b] if r['dataset']==ds and r['split']=='test' and r['true_label']==label]
    def key(r):return rank(f'{b}|{ds}|{label}|{r["sample_id"]}')
    def choose(rs):return sorted(rs,key=lambda r:(key(r),r['sample_id']))[:quotas[b]]
    assert len(eligible)>=quotas[b]
    picked=choose(eligible);assert [r['sample_id'] for r in picked]==[r['sample_id'] for r in choose(list(reversed(eligible)))]
    for r in picked:
     selected.append(dict(r,backbone=b,selection_hash=key(r),stage2_probability=quotas[b]/len(eligible),combined_inclusion_probability=r['inclusion_probability']*quotas[b]/len(eligible),probe_threshold=thresholds[b][ds]['original_threshold']))
    selection_audit.append(dict(backbone=b,dataset=ds,label=label,available=len(eligible),selected=len(picked)))
 assert len(selected)==len(datasets)*n*2
 assert all(r['split']=='test' for r in selected)
 save(ROOT/'SELECTION.json',selected);save(ROOT/'SELECTION_AUDIT.json',selection_audit)
 instruction=json.loads((SOURCE/'PROTOCOL.json').read_text())['output_instruction']
 label_instruction='Return exactly one digit: 1 for the positive class or 0 for the negative class. Do not include explanations or other text.'
 cases=[];off=[];render_audit=[];tokenizers={}
 specs=json.loads((ROOT/'EXPANSION_PLAN.json').read_text())['models']
 cache=Path('/opt/dlami/nvme/interpretable-probe-models')
 for judge in ['qwen','gemma']:
  spec=specs[judge];path=cache/('models--'+spec['repo'].replace('/','--'))/'snapshots'/spec['revision'];tokenizers[judge]=AutoTokenizer.from_pretrained(path,local_files_only=True)
 for b in BBS:
  picked=[r for r in selected if r['backbone']==b];ids={r['sample_id'] for r in picked}
  for c in CS:
   prompt_maps={}
   for j in ['qwen','gemma']:
    d=SOURCE/b/'judges'/f'{j}_{c}';p=d/'prompts.jsonl';track(p);assert sha(p)==json.loads((d/'approved_config.json').read_text())['prompt_sha256'];prompt_maps[j]={r['sample_id']:r for r in lines(p) if r['sample_id'] in ids}
    for mode,root in [('hard',PARENT),('probability',SOURCE)]:
     d=root/b/'judges'/f'{j}_{c}'/'run';p=d/'judgments.jsonl';track(p);assert sha(p)==json.loads((d/'COMPLETED').read_text())['judgments_sha256']
     for r in lines(p):
      if r['sample_id'] in ids:
       off.append(dict(id=f'{b}|{r["sample_id"]}|{c}|{mode}',backbone=b,sample_id=r['sample_id'],judge=j,condition=c,mode=mode,value=r['prediction'] if mode=='hard' else r['positive_probability'],valid=True,source=str(p)))
   for meta in picked:
    sid=meta['sample_id'];p=prompt_maps['qwen'][sid];text=p['prompt'];head='<|im_start|>system\n';sep='<|im_end|>\n<|im_start|>user\n';tail='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    assert text.startswith(head) and text.endswith(tail);system,user=text[len(head):-len(tail)].split(sep,1)
    messages=[dict(role='system',content=system),dict(role='user',content=user)]
    # Verify that canonical task text recreates BOTH old non-thinking probability prompts exactly.
    for j,tok in tokenizers.items():
     rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
     assert rendered==prompt_maps[j][sid]['prompt'],f'Non-thinking prompt mismatch: {j}/{b}/{sid}/{c}'
    render_audit.append(dict(backbone=b,sample_id=sid,condition=c,both_nonthinking_prompts_exact=True))
    for mode in ['hard','probability']:
     sy,us=system,user
     if mode=='hard':
      assert instruction in sy;sy=sy.replace(instruction,label_instruction,1);footer='Classification of the CURRENT conversation (0 or 1):' if c=='nearest' else 'Classification (0 or 1):';assert us.endswith('Positive-class probability (0-100):');us=us[:-len('Positive-class probability (0-100):')]+footer
     cases.append(dict(id=f'{b}|{sid}|{c}|{mode}',backbone=b,dataset=meta['dataset'],sample_id=sid,condition=c,mode=mode,messages=[dict(role='system',content=sy),dict(role='user',content=us)]))
 cases.sort(key=lambda r:(r['condition'],r['mode'],rank(r['backbone']+'|'+r['sample_id'])))
 assert len(cases)==len(selected)*6 and len(off)==len(cases)*2
 assert len({r['id'] for r in cases})==len(cases)
 for j in ['qwen','gemma']:assert {r['id'] for r in off if r['judge']==j}=={r['id'] for r in cases}
 (ROOT/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases));save(ROOT/'OFF_CONTROLS.json',off);save(ROOT/'PROMPT_AUDIT.json',render_audit)
 save(ROOT/'PREPARED.json',dict(base_cases=len(selected),calls_per_model=len(cases),total_new_calls=2*len(cases),cases_sha256=sha(ROOT/'cases.jsonl'),selection_sha256=sha(ROOT/'SELECTION.json'),off_controls_sha256=sha(ROOT/'OFF_CONTROLS.json'),source_hashes=source_hashes,input_reordering_test=True,exact_class_balance=True,all_target_cases_test=True,protocol_sha256=sha(ROOT/'PROTOCOL.json')))
if __name__=='__main__':main()
