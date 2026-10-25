import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from collections import namedtuple
from paper.collect import initialise,parse,run_jobs,trace_jobs,payload,request
class CollectionTests(unittest.TestCase):
 def test_parser(self):
  self.assertEqual(parse('main','Refusal to help.\n17','stop'),(17,'valid'))
  self.assertEqual(parse('main','The year is 2024','stop'),(None,'invalid'))
  self.assertEqual(parse('main','101','stop'),(None,'invalid'))
  self.assertEqual(parse('reason_after','75\nIncomplete explanation','length'),(75,'valid'))
  self.assertEqual(parse('reason_first','Incomplete\n75','length'),(None,'invalid'))
  self.assertEqual(parse('mechanism','{"labelled_similarity":"yes","important_contrast":"no"}','stop')[1],'valid')
  self.assertEqual(parse('trace_labels','{"similarity":"yes","contrast":"no","assessable":"no"}','stop')[1],'invalid')
 @patch('paper.collect.shutil.disk_usage',return_value=namedtuple('Usage','total used free')(100*2**30,50*2**30,50*2**30))
 def test_resume_never_repeats_dispatch(self,_disk):
  config={'models':[]};model={'alias':'test','model':'test','provider':'openai-compatible','url':'http://localhost','native_thinking':False}
  calls=[]
  def mock(body):calls.append(body);return {'model':'test-version','choices':[{'message':{'content':'75'},'finish_reason':'stop'}]}
  with tempfile.TemporaryDirectory() as t:
   con=initialise(Path(t)/'responses.sqlite',config)
   jobs=[{'prompt_id':str(i),'study':'main','messages':[{'role':'user','content':'test'}]} for i in range(3)]
   self.assertEqual(run_jobs(con,model,jobs,2,transport=mock),3)
   self.assertEqual(run_jobs(con,model,jobs,2,transport=mock),0)
   self.assertEqual(len(calls),3)
   con.execute('INSERT INTO requests(model,prompt_id,status) VALUES ("test","ambiguous","dispatched")');con.commit()
   self.assertEqual(run_jobs(con,model,[dict(jobs[0],prompt_id='ambiguous')],2,transport=mock),0)
   with self.assertRaises(ValueError):initialise(Path(t)/'responses.sqlite',{'models':['changed']})
 def test_malformed_raw_response_preserved(self):
  model={'provider':'openai-compatible','model':'test'}
  job={'study':'main','messages':[{'role':'user','content':'test'}]}
  raw={'unexpected':'retained'}
  result=request(model,job,transport=lambda body:raw)
  self.assertEqual(result['status'],'ambiguous')
  self.assertEqual(result['raw_json'],raw)
 def test_anthropic_payload(self):
  m={'provider':'anthropic','model':'test'};j={'study':'main','messages':[{'role':'system','content':'policy'},{'role':'user','content':'target'}]}
  p=payload(m,j);self.assertEqual(p['system'],'policy');self.assertEqual(p['max_tokens'],8);self.assertEqual(len(p['messages']),1)
if __name__=='__main__':unittest.main()
