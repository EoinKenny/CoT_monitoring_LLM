import unittest,random
from common import sample_flags,pipeline_counts
class ProtocolTests(unittest.TestCase):
 def pool(self,tp,fp):return [dict(sample_id=str(i),true_label=int(i<tp),dataset='x') for i in range(tp+fp)]
 def test_balances(self):
  for tp,fp,want in [(1000,1000,(250,250)),(700,52,(448,52)),(52,700,(52,448)),(0,700,(0,500)),(700,0,(500,0)),(20,30,(20,30)),(0,0,(0,0))]:
   rows=self.pool(tp,fp);a,m=sample_flags(rows,'b','x');random.Random(7).shuffle(rows);b,n=sample_flags(rows,'b','x');self.assertEqual((a,m),(b,n));self.assertEqual(tuple(m['strata'][s]['selected'] for s in ['TP','FP']),want)
   for y,pop in [(1,tp),(0,fp)]:self.assertAlmostEqual(sum(r['population_weight'] for r in a if r['true_label']==y),pop)
 def test_weighted_census(self):
  rows,_=sample_flags(self.pool(600,40),'b','x');base=dict(tp=600,fp=40,fn=30,tn=900)
  keep=pipeline_counts(base,rows,[1]*len(rows));reject=pipeline_counts(base,rows,[0]*len(rows))
  for k in base:self.assertAlmostEqual(keep[k],base[k])
  self.assertEqual(reject['tp'],0);self.assertEqual(reject['fp'],0);self.assertEqual(reject['fn'],630);self.assertEqual(reject['tn'],940)
 def test_identity_context(self):
  rows=self.pool(1000,1000);a,_=sample_flags(rows,'a','x');b,_=sample_flags(rows,'b','x');self.assertNotEqual([r['sample_id'] for r in a],[r['sample_id'] for r in b])
if __name__=='__main__':unittest.main()
