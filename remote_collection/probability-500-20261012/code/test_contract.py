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
 def test_valid_derived_label(self):
  for t in ['0','49','50','85','100']:
   r=parse(*self.sample(t),0,.5)[0];self.assertEqual(r['generated_text'],t);self.assertEqual(r['positive_probability'],int(t));self.assertEqual(r['prediction'],int(int(t)>=50))
 def test_invalid_never_repaired(self):
  for t in ['101','-1','1,50','050','85 extra','{"probability":50}']:
   with self.assertRaises(AssertionError):parse(*self.sample(t),0,.5)
  with self.assertRaises(AssertionError):parse(*self.sample('50','length'),0,.5)
if __name__=='__main__':unittest.main()
