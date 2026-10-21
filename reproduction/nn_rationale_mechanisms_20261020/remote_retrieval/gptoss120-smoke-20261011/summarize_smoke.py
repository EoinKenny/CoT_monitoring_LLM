from pathlib import Path
import json,re,hashlib,statistics,time,shutil
R=Path('/home/ubuntu/experiments/gptoss120-smoke-20261011');A=R/'attempt3_triton'
assert (A/'COMPLETED.json').exists()
config=json.loads((A/'CONFIG.json').read_text());prompts={r['id']:r for r in map(json.loads,(A/'prompts.jsonl').open())};batches=[]
for p in sorted(A.glob('*.result.json')):
 d=json.loads(p.read_text());raw=json.loads(p.with_name(p.name.replace('.result.json','.raw.json')).read_text());assert [r['id'] for r in raw]==[r['id'] for r in d['records']]
 parsed=[]
 for r in raw:
  source=prompts[r['id']];matches=list(re.finditer(r'<\|start\|>assistant<\|(?:channel|meta_sep)\|>final<\|message\|>',r['text']));s=r['text'][matches[-1].end():] if matches else None
  if s is not None:s=re.sub(r'<\|(?:fim_suffix|return|end)\|>$','',s).strip()
  valid=bool(s is not None and re.fullmatch(r'\d{1,3}',s) and int(s)<=(1 if source['mode']=='hard' else 100) and r['finish_reason']=='stop')
  parsed.append(dict(id=r['id'],mode=source['mode'],condition=source['condition'],input_tokens=source['input_tokens'],generated_tokens=len(r['token_ids']),final=s,valid=valid,finish_reason=r['finish_reason']))
 batches.append(dict(tag=d['tag'],seconds=d['seconds'],n=d['n'],valid=sum(r['valid'] for r in parsed),records=parsed,raw_sha256=hashlib.sha256(p.with_name(p.name.replace('.result.json','.raw.json')).read_bytes()).hexdigest()))
results=[]
for bs in [1,8,16]:
 rr=[r for r in batches if r['tag'].startswith(f'batch{bs:02d}_')];records=[v for r in rr for v in r['records']];seconds=sum(r['seconds'] for r in rr);results.append(dict(batch_size=bs,n=len(records),seconds=seconds,calls_per_second=len(records)/seconds,amortized_seconds_per_call=seconds/len(records),valid=sum(r['valid'] for r in records),mean_input_tokens=statistics.mean(r['input_tokens'] for r in records),mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in records),max_generated_tokens=max(r['generated_tokens'] for r in records)))
assert all(r['valid']==r['n'] for r in results),results
start=json.loads((A/'warmup.attempt.json').read_text())['time'];samples=[json.loads(l) for l in (A/'gpu_samples.jsonl').open()];after=[int(r['gpu'].split(',')[0]) for r in samples if r['time']>=start]
res=dict(model='openai/gpt-oss-120b',revision=config['revision'],status='passed',configuration=config,load=json.loads((A/'LOADED.json').read_text()),weights_gib=66.14,benchmarks=results,peak_gpu_mib_inference=max(after),peak_gpu_mib_including_loading=max(int(r['gpu'].split(',')[0]) for r in samples),free_root_gib=shutil.disk_usage(R).free/1024**3,free_model_volume_gib=shutil.disk_usage(config['model']).free/1024**3,parser_amendment='Original parser expected legacy meta_sep/fim_suffix spellings. Corrected post-hoc from preserved raw text to support current channel token and omitted EOS text; require explicit final channel and stop finish. No generations repeated for this parsing repair.',limitations=['Feasibility/timing smoke only, not an accuracy or F1 estimate.','48 distinct prompts span three explanations, two output formats and eight length quantiles per condition from prior smoke selection; max input 5623.','Batch 8 and 16 use the same 48 prompts; batch 1 covers six short prompts, so timing is not a matched batch-size comparison with batch 1.','Eager execution, prefix cache disabled, low reasoning. Generated-token counts include reasoning and final output; throughput is not optimized.','Two startup attempts failed before any judgments; all logs preserved. First FlashInfer attention failure, second sampler CUDA compiler-path failure. Triton attention plus disabled FlashInfer sampler worked.','Largest observed loading peak is transient; successful inference peak reported separately.'])
(R/'SMOKE_RESULTS.json').write_text(json.dumps(res,indent=2));(R/'parsed_results_v2.json').write_text(json.dumps(batches,indent=2));print(json.dumps({k:v for k,v in res.items() if k!='configuration'},indent=2))
