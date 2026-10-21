import unittest,tempfile,importlib.util,json,csv,sys,io,contextlib,hashlib
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('runner',HERE/'run_api_judges.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class RunnerTests(unittest.TestCase):
 def test_parser(self):
  for raw,label in [('0','0'),(' 1\n','1')]:self.assertEqual(mod.parse_response(raw)[:2],('ok',label))
  for raw in ['Label: 1','10','',None,0,'I cannot comply'] :self.assertEqual(mod.parse_response(raw)[0],'invalid_response')
  self.assertEqual(mod.parse_response('ERROR: unavailable')[0],'api_error')
 def test_collection_recovery_and_no_leakage(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);rows=[]
   for condition in ['baseline','saliency','nearest']:
    for i in range(26):
     sy='Classification rubric '+condition;us='Only user content '+str(i);sid='case'+str(i);h=hashlib.sha256(json.dumps([dict(role='system',content=sy),dict(role='user',content=us)],ensure_ascii=False).encode()).hexdigest();uid=hashlib.sha256(json.dumps(['b','d',condition,sid,h],separators=(',',':')).encode()).hexdigest()
     rows.append(dict(row_id=uid,schema_version='probe-api-export-v1',backbone='b',dataset='d',condition=condition,sample_id=sid,source_group=sid,system_prompt=sy,user_prompt=us,messages_sha256=h,source_prompt_sha256=h,probe_prediction='1',gold_label=str(i%2),stratum='TP' if i%2 else 'FP',selected_tp='13',selected_fp='13',population_tp='26',population_fp='26',population_weight='2',inclusion_probability='.5',test_tp='26',test_fp='26',selection_hash=sid,probe_probability='.8',probe_threshold='.7'))
   path=root/'prompts.csv'
   with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
   mod.EXPECTED_INPUT_SHA256=mod.sha(path)
   fake=root/'llm_query.py';fake.write_text('''import os,json
from pathlib import Path
model_ids={m:'fake/'+m for m in '''+repr(mod.DEFAULT_MODELS)+'''}
sys_prompt='WRONG_DEFAULT'
def call_llm(prompts,model_name,temperature):
 assert sys_prompt.startswith('Classification rubric ')
 assert len(prompts)<=15 and all(p.startswith('Only user content ') for p in prompts)
 assert temperature==0
 with Path(__file__).with_name('calls.jsonl').open('a') as f:f.write(json.dumps(dict(n=len(prompts),model=model_name,system=sys_prompt))+'\\n')
 return ['ERROR: injected' if os.getenv('FAKE_FAIL')=='yes' else ('1' if j%2 else '0') for j in range(len(prompts))]
''')
   def run(out,*extra):
    with contextlib.redirect_stdout(io.StringIO()):return mod.main(['--csv',str(path),'--module-dir',str(root),'--output',str(out),'--limit-per-model','31',*extra])
   sys.modules.pop('llm_query',None);out=root/'success'
   self.assertEqual(run(out),0);rr=list(csv.DictReader((out/'results.csv').open()));self.assertEqual(len(rr),279);self.assertEqual(len({(r['api_model'],r['row_id']) for r in rr}),279)
   calls=(root/'calls.jsonl').read_text();self.assertEqual(len(calls.splitlines()),27)
   self.assertEqual(run(out),0);self.assertEqual((root/'calls.jsonl').read_text(),calls)
   # Lost convenience export and missing parsed checkpoint recover from preserved raw data.
   next((out/'state').glob('*.done.json')).unlink();(out/'results.csv').write_text('interrupted CSV')
   self.assertEqual(run(out),0);self.assertEqual((root/'calls.jsonl').read_text(),calls);self.assertEqual(len(list(csv.DictReader((out/'results.csv').open()))),279)
   # Simulate a lost raw+parsed result, retaining intent: never silently issue it again.
   p=next((out/'state').glob('*.done.json'));key=p.name[:-len('.done.json')];p.unlink();(out/'state'/(key+'.returned.json')).unlink()
   self.assertEqual(run(out),2);self.assertEqual((root/'calls.jsonl').read_text(),calls)
   rr=list(csv.DictReader((out/'results.csv').open()));self.assertTrue(any(r['status']=='unresolved' for r in rr));self.assertEqual(len(rr),279);self.assertFalse((out/'COMPLETED.json').exists())
   # API error stops this model while retaining its attempted rows; no retry on resume.
   import os
   os.environ['FAKE_FAIL']='yes';failed=root/'failed';self.assertEqual(run(failed,'--models','gpt5'),2);before=(root/'calls.jsonl').read_text();self.assertEqual(run(failed,'--models','gpt5'),2);self.assertEqual((root/'calls.jsonl').read_text(),before)
   del os.environ['FAKE_FAIL'];self.assertEqual(run(failed,'--models','gpt5','--continue-blocked','--recovery-note','Mock endpoint fixed; continue untouched cases'),2)
   rr=list(csv.DictReader((failed/'results.csv').open()));self.assertEqual(len(rr),31);self.assertEqual(sum(r['status']=='api_error' for r in rr),15)
   sys.modules.pop('llm_query',None)
if __name__=='__main__':unittest.main()
