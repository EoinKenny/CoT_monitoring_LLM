import unittest,tempfile
from setup import transform,OLD_INSTRUCTION,INSTRUCTION,FOOTER
from run_judge import parse
class Tests(unittest.TestCase):
 def test_contract_preserves_task_and_payload(self):
  for c in ['baseline','saliency','nearest']:
   foot='Classification of the CURRENT conversation (0 or 1):' if c=='nearest' else 'Classification (0 or 1):'
   x=dict(prompt_id='old',sample_id='id',backbone='llama',condition=c,messages=[dict(role='system',content='prefix '+OLD_INSTRUCTION+' Task: preserved'),dict(role='user',content='data\nexample label: 1\n'+foot)])
   a=transform(x);self.assertEqual(a,transform(x));self.assertEqual(a['messages'][1]['content'],'data\nexample label: 1\n'+FOOTER);self.assertIn(INSTRUCTION,a['messages'][0]['content']);self.assertNotEqual(a['prompt_id'],'old');self.assertNotEqual(a['prompt_id'],transform(dict(x,sample_id='another'))['prompt_id'])
 def sample(self,text,finish='stop'):
  p=dict(sample_id='id',prompt_id='pid',dataset='d',backbone='b',condition='baseline',input_tokens=1,instruction_sha256='hash',token_ids=[1])
  raw=[dict(prompt_token_ids=[1],outputs=[dict(text=text,token_ids=[2],finish_reason=finish)])]
  return raw,[p]
 def test_valid_and_disagreement_retained(self):
  for t in ['0,100','1,0','1,85','0,0','1,100']:
   r=parse(*self.sample(t),0,.5)[0];self.assertEqual(r['generated_text'],t);self.assertEqual((r['prediction'],r['positive_probability']),tuple(map(int,t.split(','))))
 def test_invalid_never_repaired(self):
  for t in ['1,101','2,50','1, 50','1,050','1,50 extra','{"label":1,"probability":50}']:
   with self.assertRaises(AssertionError):parse(*self.sample(t),0,.5)
  with self.assertRaises(AssertionError):parse(*self.sample('1,50','length'),0,.5)
if __name__=='__main__':unittest.main()
