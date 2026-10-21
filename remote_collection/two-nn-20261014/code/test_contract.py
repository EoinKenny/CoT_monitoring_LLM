import unittest,numpy as np
from setup import best_other,transform,INTRO,FOOTER
from run_judge import parse
class Tests(unittest.TestCase):
 def test_exclude_and_stable_ties(self):
  T=np.array([[1.,0],[1.,0],[0.,1],[-1.,0]])
  Z=np.array([[1.,0],[1.,0],[0.,1]])
  a,s,t,_=best_other(Z,T,np.array([0,1,2]),chunk=2)
  self.assertEqual(a.tolist(),[1,0,0]);self.assertEqual(t.tolist(),[0,0,2])
  b,*_=best_other(Z[::-1],T,np.array([2,1,0]),chunk=2);self.assertEqual(b[::-1].tolist(),a.tolist())
 def test_prompt_preserves_first(self):
  original=[dict(role='system',content=INTRO+"\nThe reference example includes its dataset label and the probe's prediction. Similarity does not guarantee that the reference and the current conversation belong to the same class. Use the reference as supporting context.\nReturn only one integer 0 to 100"),dict(role='user',content='TARGET\nFIRST REFERENCE\n\n'+FOOTER)]
  n=dict(messages=[dict(role='user',content='SECOND')],label=0,prediction=1)
  x=transform(original,n,{0:'negative',1:'positive'})
  self.assertTrue(x[1]['content'].startswith(original[1]['content'][:-len(FOOTER)]));self.assertTrue(x[1]['content'].endswith(FOOTER));self.assertIn('Dataset label: negative (0)',x[1]['content']);self.assertIn('Probe prediction: positive (1)',x[1]['content']);self.assertIn('two most similar distinct',x[0]['content'])
 def test_probability(self):
  p=dict(token_ids=[1],sample_id='s',prompt_id='p',dataset='d',backbone='b',condition='nearest2',input_tokens=1,instruction_sha256='h')
  def go(s):return parse([dict(prompt_token_ids=[1],outputs=[dict(text=s,token_ids=[2],finish_reason='stop')])],[p],0,.1)[0]
  for n in [0,49,50,100]:self.assertEqual(go(str(n))['prediction'],int(n>=50))
  for s in ['101','85%','reasoning 85','-1']:
   with self.assertRaises(AssertionError):go(s)
if __name__=='__main__':unittest.main()
