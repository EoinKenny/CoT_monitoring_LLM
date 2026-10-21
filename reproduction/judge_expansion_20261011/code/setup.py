from common import *
import shutil
def main():
 reserve();assert not (ROOT/'SETUP_COMPLETED.json').exists()
 sources={}
 for mode,src in [('hard',HARD),('probability',PROB)]:
  assert sha(src/'RESULTS.json')==read(src/'COMPLETED.json')['results_sha256']
  sources[str(src/'RESULTS.json')]=sha(src/'RESULTS.json')
  for b in BBS:
   out=ROOT/mode/b;out.mkdir(parents=True,exist_ok=True)
   for name in ['selected_flags.json','SAMPLE_FROZEN.json','THRESHOLDS.json']:
    p=src/b/name;sources[str(p)]=sha(p);shutil.copy2(p,out/name)
   assert sha(out/'selected_flags.json')==read(out/'SAMPLE_FROZEN.json')['selection_sha256']
   ids=[r['sample_id'] for r in read(out/'selected_flags.json')]
   assert len(ids)==len(set(ids))==10000 and ids==read(out/'SAMPLE_FROZEN.json')['sample_ids']
   assert ids==[r['sample_id'] for r in read(HARD/b/'selected_flags.json')]
   for p in (src/b).glob('*_predictions.npz'):(out/p.name).symlink_to(p.resolve())
   (out/'judge_inputs').symlink_to(HARD/b/'judge_inputs',target_is_directory=True)
   (out/'judges').mkdir(exist_ok=True)
   for j in ['qwen','gemma']:
    for c in CS:
     d=src/b/'judges'/f'{j}_{c}';done=read(d/'run/COMPLETED')
     assert sha(d/'run/judgments.jsonl')==done['judgments_sha256']
     assert sha(d/'run/config.json')==done['config_sha256']
     (out/'judges'/d.name).symlink_to(d,target_is_directory=True)
     sources[str(d/'run/judgments.jsonl')]=done['judgments_sha256']
 write(ROOT/'SETUP_COMPLETED.json',dict(time=time.time(),sources=sources,existing_judgments=600000,new_judgments=1500000,expected_total=2100000))
if __name__=='__main__':main()
