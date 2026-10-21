import unittest
from setup import transform,random_index,REFERENCE_START,USE,FOOTER,FIELDS
from run_judge import parse
class Tests(unittest.TestCase):
 def test_ablations(self):
  messages=[dict(role='system',content='TASK\n\n'+REFERENCE_START+'\n\nThe reference example includes its dataset label and the probe\'s prediction. The probe may disagree with the dataset label. Similarity does not guarantee that the reference and the current conversation belong to the same class.\n\n'+USE+'\n\nReturn only one integer from 0 to 100'),dict(role='user',content='Classify this supplied conversation JSON:\n[{"role":"user","content":"TARGET"}]\n\nNearest-neighbour explanation:\nThe most similar training example was:\n[{"role":"user","content":"NEIGHBOUR"}]\n\nReference example:\nDataset label: positive (1) |\nProbe prediction: negative (0)\n\n'+FOOTER)]
  random=dict(messages=[dict(role='user',content='RANDOM')],label=0,prediction=1)
  for c,(t,l,p) in FIELDS.items():
   out=transform(messages,c,random,{0:'negative',1:'positive'});user=out[1]['content'];self.assertTrue(user.endswith(FOOTER));self.assertIn('TARGET',user)
   self.assertEqual('Dataset label:' in user,bool(l));self.assertEqual('Probe prediction:' in user,bool(p))
   if c=='random_full':self.assertIn('RANDOM',user);self.assertNotIn('NEIGHBOUR',user)
   else:self.assertEqual('NEIGHBOUR' in user,bool(t))
   self.assertTrue(out[0]['content'].startswith('TASK\n\n'));self.assertTrue(out[0]['content'].endswith('Return only one integer from 0 to 100'))
 def test_random_reordering(self):
  ids=['b','a','c'];a={i:random_index('ds',i,57) for i in ids};b={i:random_index('ds',i,57) for i in reversed(ids)};self.assertEqual(a,b);self.assertTrue(all(0<=v[0]<57 for v in a.values()))
 def test_probability(self):
  p=dict(token_ids=[1],sample_id='x',prompt_id='p',dataset='d',backbone='gemma',condition='text_only',input_tokens=1,instruction_sha256='h')
  def go(s):return parse([dict(prompt_token_ids=[1],outputs=[dict(text=s,token_ids=[2],finish_reason='stop')])],[p],0,.1)[0]
  for n in [0,49,50,100]:self.assertEqual(go(str(n))['prediction'],int(n>=50))
  for bad in ['101','-1','85%','Reasoning 85','{"probability":85}']:
   with self.assertRaises(AssertionError):go(bad)
if __name__=='__main__':unittest.main()
