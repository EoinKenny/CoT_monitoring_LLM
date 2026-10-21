import unittest,tempfile,json
from pathlib import Path
from imports import create_imports,verify_imports
from common import write,sha,digest
class ImportTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name);self.src=self.r/'source';self.dst=self.r/'dest'
  self.dst.mkdir();(self.dst/'prompts.jsonl').write_text(json.dumps({'sample_id':'a'})+'\n')
  write(self.src/'run/config.json',{'fixed':True});raw=self.src/'run/batches/000000.raw.json';write(raw,[{'text':'1'},{'text':'0'}])
  rr=[dict(sample_id='a',prediction=1),dict(sample_id='b',prediction=0)]
  write(self.src/'run/batches/000000.json',dict(records=rr,records_sha256=digest(json.dumps(rr,sort_keys=True)),config_sha256=sha(self.src/'run/config.json'),raw_sha256=sha(raw)))
 def tearDown(self):self.tmp.cleanup()
 def test_subset_retains_original_source(self):
  self.assertEqual(create_imports(self.src,self.dst,{'a'}),[dict(sample_id='a',prediction=1)])
  self.assertEqual(len(json.loads((self.src/'run/batches/000000.json').read_text())['records']),2)
 def test_raw_tampering_rejected(self):
  create_imports(self.src,self.dst,{'a'});write(self.src/'run/batches/000000.raw.json',[{'text':'0'}])
  with self.assertRaises(AssertionError):verify_imports(self.dst/'run/IMPORTED.json')
 def test_record_tampering_rejected(self):
  create_imports(self.src,self.dst,{'a'});p=self.dst/'run/IMPORTED.json';x=json.loads(p.read_text());x['records_sha256']='wrong';write(p,x)
  with self.assertRaises(AssertionError):verify_imports(p)
 def test_nested_projection_retains_original_proof(self):
  create_imports(self.src,self.dst,{'a'})
  write(self.dst/'run/config.json',{'projection':True})
  target=self.r/'nested';target.mkdir();(target/'prompts.jsonl').write_text(json.dumps({'sample_id':'a'})+'\n')
  rows=create_imports(self.dst,target,{'a'})
  self.assertEqual(rows,[dict(sample_id='a',prediction=1)])
  proof=json.loads((target/'run/IMPORTED.json').read_text())
  self.assertEqual(proof['batches'][0]['source_batch'],str(self.src/'run/batches/000000.json'))
if __name__=='__main__':unittest.main()
