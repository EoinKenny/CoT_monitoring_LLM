from pathlib import Path
import json,hashlib
R=Path('/home/ubuntu/experiments/reasoning-benchmark-two-batches-20261011')
Q=Path('/home/ubuntu/experiments/reasoning-benchmark-quick-20261011')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
qs=json.loads((Q/'SUMMARY.json').read_text())['models']['qwen']
out={'qwen':qs}
cases=[json.loads(l) for l in (R/'cases.jsonl').open()]
assert sha(R/'cases.jsonl')==json.loads((R/'PREPARED.json').read_text())['cases_sha256']
for model in ['gemma','glm','gptoss']:
 d=R/model;done=json.loads((d/'COMPLETED.json').read_text());assert sha(d/'RESULTS.json')==done['results_sha256']
 cfg=json.loads((d/'CONFIG.json').read_text());assert sha(d/'prompts.jsonl')==cfg['prompt_sha256']
 x=json.loads((d/'RESULTS.json').read_text());bs=x['batches'];assert len(bs)==2 and all(b['n']==16 for b in bs)
 rr=[]
 for b in bs:
  assert sha(d/(b['tag']+'.raw.json'))==b['raw_sha256'];rr+=b['records']
 assert len(rr)==32 and {r['id'] for r in rr}=={r['id'] for r in cases}
 hours={b['records'][0]['mode']:b['seconds']/16*150000/3600 for b in bs}
 out[model]=dict(n=32,valid=sum(r['valid'] for r in rr),truncated=sum(r['finish_reason']=='length' for r in rr),reasoning_delimiter=sum(r['reasoning_delimiter'] for r in rr),hard_hours=hours['hard'],probability_hours=hours['probability'],full_hours=sum(hours.values()),load_seconds=x['load_seconds'],batch_seconds=[b['seconds'] for b in bs],mean_generated_tokens=sum(r['generated_tokens'] for r in rr)/32,max_generated_tokens=max(r['generated_tokens'] for r in rr),mean_input_tokens=sum(r['input_tokens'] for r in rr)/32,config_sha256=sha(d/'CONFIG.json'),results_sha256=sha(d/'RESULTS.json'))
qdone=json.loads((Q/'qwen/COMPLETED.json').read_text());assert sha(Q/'qwen/RESULTS.json')==qdone['results_sha256']
report=dict(models=out,total_hours=sum(x['full_hours'] for x in out.values()),calls_per_model=300000,formula='150000 hard-label judgments +150000 probability judgments/model. Five backbones x ten datasets x1000 frozen validation/test cases x three explanation conditions x two output formats.',limits=['Two batches16/model as requested; Qwen300 already completed before reduction and is reused without new calls.','Qwen estimates use full-sized batches only. Other models have one hard and one probability batch, each mixing all3explanations. Estimates are coarse, not precise ETAs or validated completion guarantees.','All10datasets and5backbones appear in the reduced sample, but counts are not balanced across them; rare long cases are poorly estimated by32 calls.','Same GPU, eager inference, no prefix cache, native thinking enabled, 2048 total generated-token cap. GPT uses low reasoning effort; other models native thinking.','Model-loading time excluded; first-generation warmup may remain in32-call estimates. Native final answers parsed without final-only constrained decoding.','Cutoffs count as incomplete. Recovering these would require a separately authorized/configured run; these timings do not guarantee every case gets a valid label.','No full research collection launched. Preliminary and interrupted outputs remain preserved.'])
(R/'FINAL_SUMMARY.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
