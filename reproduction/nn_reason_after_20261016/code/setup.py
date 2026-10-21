from common import *
import shutil

def transform(r,old,new):
 assert r['condition']=='nearest' and len(r['messages'])==2
 msgs=[dict(x) for x in r['messages']];s=msgs[0]['content']
 assert s.endswith(old) and s.count(old)==1
 msgs[0]['content']=s[:-len(old)]+new
 assert msgs[1]==r['messages'][1] and msgs[1]['content'].endswith('Positive-class probability (0-100):')
 mh=digest(json.dumps(msgs,ensure_ascii=False,separators=(',',':')))
 return dict(r,condition=CS[0],messages=msgs,messages_sha256=mh,parent_prompt_id=r['prompt_id'],prompt_id='nra_'+digest(json.dumps([r['backbone'],r['sample_id'],mh],separators=(',',':'))))

def main():
 reserve();assert not (ROOT/'SETUP_COMPLETED.json').exists()
 assert read(PARENT/'COMPLETED.json')['n']==900000
 src=PARENT/'canonical/cases.jsonl';assert sha(src)==read(PARENT/'canonical/PREPARED.json')['cases_sha256']
 old=(ROOT/'OLD_OUTPUT.txt').read_text();new=(ROOT/'NEW_OUTPUT.txt').read_text()
 write(ROOT/'judge_specs.json',read(PARENT/'judge_specs.json'))
 for b in BBS:
  d=ROOT/'score'/b;d.mkdir(parents=True,exist_ok=True)
  for name in ['selected_flags.json','SAMPLE_FROZEN.json','THRESHOLDS.json']:
   shutil.copyfile(PARENT/'score'/b/name,d/name)
  for p in (PARENT/'score'/b).glob('*_predictions.npz'):(d/p.name).symlink_to(p.resolve())
 dest=ROOT/'canonical/cases.jsonl';dest.parent.mkdir(exist_ok=True);assert not dest.exists()
 ids=[];by={b:[] for b in BBS};examples={}
 with dest.with_suffix('.tmp').open('w') as f:
  for r in jl(src):
   if r['condition']!='nearest':continue
   x=transform(r,old,new);f.write(json.dumps(x,ensure_ascii=False)+'\n');ids.append(x['prompt_id']);by[x['backbone']].append(x['sample_id']);examples.setdefault(x['dataset'],x['messages'])
  f.flush();os.fsync(f.fileno())
 assert len(ids)==len(set(ids))==50000
 for b in BBS:assert by[b]==[x['sample_id'] for x in read(ROOT/'score'/b/'selected_flags.json')]
 dest.with_suffix('.tmp').replace(dest)
 proof=dict(n=50000,cases_sha256=sha(dest),parent_sha256=sha(src),exact_user_messages=True,only_output_instruction_changed=True,time=time.time())
 write(ROOT/'canonical/PREPARED.json',proof);write(ROOT/'PROMPT_EXAMPLES.json',examples);write(ROOT/'SETUP_COMPLETED.json',proof)
if __name__=='__main__':main()
