from common import *
import csv,collections,zipfile
OLD=Path('/home/ubuntu/experiments/judge-expansion-20261011')
INSTRUCTION='Return only one integer from 0 to 100 estimating the probability\nthat the CURRENT conversation belongs to the positive class.\n\n0 means certainly negative.\n50 means equally likely positive or negative.\n100 means certainly positive.\n\nA value of 50 or above will be classified as positive;\na value below 50 will be classified as negative.\n\nReport the probability of the positive class, not confidence\nin whichever class you predict. Do not include any other text.'
OLD_INSTRUCTION='Return exactly one digit: 1 for the positive class or 0 for the negative class. Do not include explanations or other text.'
FOOTER='Positive-class probability (0-100):'
def transform(r):
 r=dict(r);messages=[dict(m) for m in r['messages']];assert len(messages)==2 and messages[0]['role']=='system' and messages[1]['role']=='user'
 assert messages[0]['content'].count(OLD_INSTRUCTION)==1
 messages[0]['content']=messages[0]['content'].replace(OLD_INSTRUCTION,INSTRUCTION,1)
 oldfooter='Classification of the CURRENT conversation (0 or 1):' if r['condition']=='nearest' else 'Classification (0 or 1):'
 text=messages[1]['content'];i=text.rindex(oldfooter);messages[1]['content']=text[:i]+FOOTER+text[i+len(oldfooter):]
 # Prove all task/explanation/example bytes remain unchanged.
 assert messages[0]['content'].replace(INSTRUCTION,OLD_INSTRUCTION,1)==r['messages'][0]['content']
 assert messages[1]['content'][:i]+oldfooter+messages[1]['content'][i+len(FOOTER):]==text
 mid=digest(json.dumps(messages,sort_keys=True,ensure_ascii=False));pid='s50_'+digest(json.dumps([r['backbone'],r['sample_id'],r['condition'],'score',mid],ensure_ascii=False))
 return dict(r,parent_prompt_id=r['prompt_id'],prompt_id=pid,id=pid,mode='score',messages=messages,messages_sha256=mid)
def main():
 reserve();assert not (ROOT/'SETUP_COMPLETED.json').exists();source=OLD/'canonical/cases.jsonl';assert sha(source)==read(OLD/'canonical/PREPARED.json')['cases_sha256']
 expected={};sourcehash={str(source):sha(source)}
 for b in BBS:
  src=HARD/b;rows=read(src/'selected_flags.json');assert len(rows)==10000 and sha(src/'selected_flags.json')==read(src/'SAMPLE_FROZEN.json')['selection_sha256'];assert not {(r['dataset'],r['group']) for r in rows if r['split']=='val'}&{(r['dataset'],r['group']) for r in rows if r['split']=='test'}
  assert set(collections.Counter((r['dataset'],r['split']) for r in rows).values())=={500}
  expected[b]=[r['sample_id'] for r in rows];dest=ROOT/'score'/b;dest.mkdir(parents=True)
  for name in ['selected_flags.json','SAMPLE_FROZEN.json','THRESHOLDS.json']:
   shutil.copy2(src/name,dest/name);sourcehash[str(src/name)]=sha(src/name)
  for p in src.glob('*_predictions.npz'):(dest/p.name).symlink_to(p.resolve())
  (dest/'judge_inputs').symlink_to(src/'judge_inputs',target_is_directory=True)
  for mode in ['hard','probability']:
   view=ROOT/mode/b;view.mkdir(parents=True)
   for name in ['selected_flags.json','SAMPLE_FROZEN.json','THRESHOLDS.json','judge_inputs']+[p.name for p in dest.glob('*_predictions.npz')]:
    (view/name).symlink_to(dest/name,target_is_directory=(name=='judge_inputs'))
   (dest/'judges').mkdir(exist_ok=True);(view/'judges').symlink_to(dest/'judges',target_is_directory=True)
 canonical=ROOT/'canonical';canonical.mkdir();out=ROOT/'api_export';ids=[];by=collections.defaultdict(list);balances=collections.Counter()
 with (canonical/'cases.jsonl').open('w') as f,(out/'prompts.csv').open('w',newline='') as cf:
  w=csv.DictWriter(cf,fieldnames=['prompt_id','output_type','system_prompt','user_prompt'],lineterminator='\n');w.writeheader()
  for r in jl(source):
   if r['mode']!='hard':continue
   r=transform(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');w.writerow(dict(prompt_id=r['prompt_id'],output_type='probability',system_prompt=r['messages'][0]['content'],user_prompt=r['messages'][1]['content']));ids.append(r['prompt_id']);by[r['backbone'],r['condition']].append(r['sample_id']);balances[r['backbone'],r['dataset'],r['split'],r['condition']]+=1
 assert len(ids)==len(set(ids))==150000 and len(balances)==300 and set(balances.values())=={500}
 for (b,c),si in by.items():assert si==expected[b]
 csv.field_size_limit(100000000)
 with (out/'prompts.csv').open(newline='') as f:
  for a,b in zip(csv.DictReader(f),jl(canonical/'cases.jsonl'),strict=True):assert a==dict(prompt_id=b['prompt_id'],output_type='probability',system_prompt=b['messages'][0]['content'],user_prompt=b['messages'][1]['content'])
 write(canonical/'PREPARED.json',dict(total=150000,cases_sha256=sha(canonical/'cases.jsonl'),source_hashes=sourcehash))
 write(out/'export_manifest.json',dict(total=150000,prompts_csv_sha256=sha(out/'prompts.csv'),ordered_ids_sha256=digest('\n'.join(ids)+'\n'),output_type='probability',per_split_per_pair=500,no_target_gold_exported=True,original_explanations_preserved=True))
 template=read(out/'model_manifest.template.json');template['prompts_csv_sha256']=sha(out/'prompts.csv');write(out/'model_manifest.template.json',template)
 names=['README.txt','prompts.csv','responses_template.csv','model_manifest.template.json','validate_results.py','export_manifest.json'];manifest=read(out/'export_manifest.json');manifest['files']={n:sha(out/n) for n in names if n!='export_manifest.json'};write(out/'export_manifest.json',manifest)
 zpath=out/'external_api_probability_150000_20261012.zip'
 with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n in names:z.write(out/n,n)
 with zipfile.ZipFile(zpath) as z:
  assert z.testzip() is None
  for n in names:
   h=hashlib.sha256()
   with z.open(n) as f:
    for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
   assert h.hexdigest()==sha(out/n)
 write(out/'PACKAGE_COMPLETED.json',dict(rows=150000,archive=str(zpath),archive_bytes=zpath.stat().st_size,archive_sha256=sha(zpath)))
 write(ROOT/'SETUP_COMPLETED.json',dict(time=time.time(),prompts=150000,calls=900000,source_hashes=sourcehash));print(read(out/'PACKAGE_COMPLETED.json'),flush=True)
if __name__=='__main__':main()
