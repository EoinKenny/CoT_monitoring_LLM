from common import *
# Estimate generation from previously observed batch timings, excluding reused records.
lookup={'llama':E/'ten-dataset-sampled-20261002','qwen':OLD['qwen'],'gemma':OLD['gemma']}
rows=[]
for b in OLD:
 for m in JUDGES:
  for c in CONDITIONS:
   cfg=read(ROOT/b/'judges'/f'{m}_{c}/approved_config.json');candidates=[lookup[b]/'judges'/f'{m}_{c}/run',E/'expanded-judges-20261003'/b/'judges'/f'{m}_{c}/run',lookup['llama']/'judges'/f'{m}_{c}/run',E/'expanded-judges-20261003/llama/judges'/f'{m}_{c}/run']
   historical=None
   for run in candidates:
    p=run/'judgments.jsonl'
    if not (run/'COMPLETED').exists() or not p.exists():continue
    records=[json.loads(l) for l in p.open()];batches={}
    for r in records:
     if 'reused_from' in r or 'batch_offset' not in r or 'batch_seconds' not in r:continue
     key=r['batch_offset'];item=batches.setdefault(key,dict(seconds=0.,tokens=0));item['seconds']=max(item['seconds'],r['batch_seconds']);item['tokens']+=r['input_tokens']
    if batches:
     seconds=sum(x['seconds'] for x in batches.values());tokens=sum(x['tokens'] for x in batches.values());historical=dict(path=str(run),generation_seconds=seconds,input_tokens=tokens,seconds_per_token=seconds/tokens);break
   rows.append(dict(backbone=b,judge=m,condition=c,cases=cfg['n'],input_tokens=cfg['total_input_tokens'],historical=historical,estimated_seconds=cfg['total_input_tokens']*historical['seconds_per_token']+240 if historical else None))
known=[r['estimated_seconds'] for r in rows if r['estimated_seconds'] is not None]
write(ROOT/'ETA.json',dict(time=time.time(),total_judgments=sum(r['cases'] for r in rows),estimated_inference_hours=sum(known)/3600 if len(known)==45 else None,conditions_with_reference=len(known),conditions=rows,method='Historical measured generation seconds per input token, condition/model/backbone when available, plus four minutes model-load allowance per condition. Approximate: new length distributions and quantization startup affect throughput. No reuse assumed.'))
print(json.dumps({k:v for k,v in read(ROOT/'ETA.json').items() if k!='conditions'}),flush=True)
