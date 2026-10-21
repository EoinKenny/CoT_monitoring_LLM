from pathlib import Path
import json,gzip,hashlib,collections,re,time,numpy as np
ROOT=Path('/home/ubuntu/experiments/nn-mechanisms-20261015');ROOT.mkdir(exist_ok=True)
P=ROOT.parent/'probability-500-20261012';H=ROOT.parent/'full-threshold-sweep-20261009'
BBS=['llama','qwen','gemma','llama70','qwen122'];sources={};stats=[];out=[]
def read(p):
 raw=p.read_bytes();sources[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw));return json.loads(raw)
def text(msgs):return '\n'.join(str(m.get('content','')) for m in msgs)
def words(t):return re.findall(r'\w+',t.lower())
for b in BBS:
 selected=read(P/'score'/b/'selected_flags.json');need={r['sample_id'] for r in selected}
 ep=H/b/'judge_inputs/explanations.jsonl';raw=ep.read_bytes();eh=hashlib.sha256(raw).hexdigest();sources[str(ep)]=dict(sha256=eh,bytes=len(raw))
 marker=H/b/'judge_inputs/EXTRACTION_COMPLETED'
 if marker.exists():
  done=read(marker)
  if 'explanations_sha256' in done:assert done['explanations_sha256']==eh
 refs={r['sample_id']:r for r in map(json.loads,raw.splitlines()) if r['sample_id'] in need};del raw
 assert len(refs)==len(selected)==10000
 for ds in sorted({r['dataset'] for r in selected}):
  meta=read(H/b/f'{ds}_metadata.json');counts=collections.Counter((r['split'],int(r['label'])) for r in meta)
  for split in sorted({r['split'] for r in meta}):
   rows=[r for r in meta if r['split']==split];lens=[len(text(r['messages'])) for r in rows]
   stats.append(dict(backbone=b,dataset=ds,split=split,n=len(rows),positive=sum(int(r['label']) for r in rows),negative=sum(1-int(r['label']) for r in rows),unique_groups=len({str(r.get('group',r.get('input_hash',r['id']))) for r in rows}),mean_chars=float(np.mean(lens)),median_chars=float(np.median(lens))))
  for r in selected:
   if r['dataset']!=ds:continue
   ex=refs[r['sample_id']];n=ex['nearest'];target=meta[r['index']];nn=meta[n['index']]
   assert target['split']==r['split'] and int(target['label'])==r['true_label']
   assert nn['split']=='train' and int(nn['label'])==n['label']
   assert target['messages']==ex['messages'] and nn['messages']==n['messages']
   assert str(target['group'])!=str(nn['group'])
   tt=text(ex['messages']);nt=text(n['messages']);tw=words(tt);nw=words(nt);a=set(tw);z=set(nw)
   out.append(dict(backbone=b,**r,nearest_id=nn.get('immutable_id',ds+':'+str(nn.get('input_hash',nn['id']))),nearest_label=n['label'],nearest_prediction=n['prediction'],nearest_cosine=n['cosine'],target_chars=len(tt),target_words=len(tw),neighbor_chars=len(nt),neighbor_words=len(nw),target_message_count=len(ex['messages']),neighbor_message_count=len(n['messages']),lexical_jaccard=len(a&z)/len(a|z) if a|z else 1.,lexical_target_coverage=len(a&z)/len(a) if a else 1.,exact_text_match=tt==nt,target_content_hash=hashlib.sha256(tt.encode()).hexdigest(),neighbor_content_hash=hashlib.sha256(nt.encode()).hexdigest(),target_tokens=target.get('n_tokens',target.get('all_tokens',len(target.get('token_ids',[])))),target_selected_tokens=target.get('selected_tokens'),pooling_error=ex.get('pooling_error'),target_messages=ex['messages'],neighbor_messages=n['messages'],neighbor_target_label_agree=n['label']==r['true_label'],neighbor_label_prediction_agree=n['label']==n['prediction']))
  print('done',b,ds,flush=True);del meta
with gzip.open(ROOT/'features.jsonl.gz','wt') as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
(ROOT/'dataset_stats.json').write_text(json.dumps(stats,indent=2))
# Rubrics are frozen task definitions, not model instructions for this analysis.
rubric=H/'llama/judge_rubrics.json'
if rubric.exists():(ROOT/'task_rubrics.json').write_text(json.dumps(read(rubric),indent=2))
manifest=dict(time=time.time(),n=len(out),sources=sources,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.name in ['features.jsonl.gz','dataset_stats.json','task_rubrics.json']})
(ROOT/'EXPORT_COMPLETED.json').write_text(json.dumps(manifest,indent=2));print('EXPORTED',len(out))
