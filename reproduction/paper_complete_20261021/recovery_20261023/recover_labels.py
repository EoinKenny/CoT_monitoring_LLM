import json,re,hashlib,pathlib,collections,os,time
ROOT=pathlib.Path('/home/ubuntu/experiments/paper-complete-20261021')
DEST=ROOT/'recovery_20261023'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def parse(text,finish):
 if finish!='stop':return None,'truncated' if finish=='length' else 'invalid'
 s=text.strip();m=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',s,re.S)
 if m:s=m.group(1).strip()
 try:
  x=json.loads(s)
  assert isinstance(x,dict) and set(x)=={'labelled_similarity','important_contrast'} and all(v in ['yes','no','uncertain'] for v in x.values())
  return x,'valid'
 except Exception:return None,'invalid'
def main():
 src=ROOT/'runs/mechanism/all_glm';out=DEST/'glm_mechanism_reparsed.jsonl';receipt=DEST/'GLM_RECOVERED.json'
 assert not out.exists() and not receipt.exists()
 completion=json.loads((src/'COMPLETED.json').read_text());assert sha(src/'responses.jsonl')==completion['responses_sha256']
 batch_records={};batches=0
 for f in sorted((src/'batches').glob('*.json')):
  if f.name.endswith(('.raw.json','.attempt.json')):continue
  b=json.loads(f.read_text());raw=f.with_suffix('.raw.json');assert sha(raw)==b['raw_sha256']
  rawrows=json.loads(raw.read_text());assert len(rawrows)==len(b['records'])
  for r,rr in zip(b['records'],rawrows):
   assert r['prompt_id']==rr['prompt_id'] and r['raw_response']==rr['outputs'][0]['text'] and r['finish_reason']==rr['outputs'][0]['finish_reason']
   assert r['prompt_id'] not in batch_records;batch_records[r['prompt_id']]=r
  batches+=1
 counts=collections.Counter();old=collections.Counter();n=0
 with out.open('x') as f, (src/'responses.jsonl').open() as h:
  for line in h:
   r=json.loads(line);assert batch_records.pop(r['prompt_id'])==r
   old[r['status']]+=1;r['original_status']=r['status'];r['original_value']=r['value'];r['value'],r['status']=parse(r['raw_response'],r['finish_reason']);counts[r['status']]+=1
   r['parser_revision']='full_json_or_single_markdown_json_fence_v1';f.write(json.dumps(r,ensure_ascii=False)+'\n');n+=1
  f.flush();os.fsync(f.fileno())
 assert not batch_records and n==completion['n']==25000
 info=dict(n=n,original_statuses=dict(old),statuses=dict(counts),source=str(src/'responses.jsonl'),source_sha256=sha(src/'responses.jsonl'),output=str(out),output_sha256=sha(out),verified_raw_batches=batches,parser_sha256=sha(__file__),rule='Only remove one complete enclosing Markdown JSON fence; require normal stop, exact keys, allowed values. No inference or raw-output changes.',time=time.time())
 receipt.write_text(json.dumps(info,indent=2));print(json.dumps(info))
if __name__=='__main__':main()
