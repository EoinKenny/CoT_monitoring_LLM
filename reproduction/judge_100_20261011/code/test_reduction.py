import unittest,random,math
from reduction import select
class SelectionTests(unittest.TestCase):
 def pool(self,counts):
  return [dict(dataset='task',split='test',stratum=f'test:{s}:0',sample_id=f'{s}/{i}',selection_hash='old',inclusion_probability=.25) for s,n in enumerate(counts) for i in range(n)]
 def test_reordering_and_weights(self):
  rows=self.pool([25]*20);x,a=select('b',rows);random.Random(17).shuffle(rows)
  self.assertEqual((x,a),select('b',rows));self.assertEqual(len(x),100)
  self.assertEqual(sorted(z['selected_n'] for z in a.values()),[5]*20)
  for k,z in a.items():
   rr=[r for r in x.values() if r['stratum']==k]
   self.assertAlmostEqual(sum(1/r['inclusion_probability'] for r in rr),z['population'])
 def test_shortage(self):
  x,a=select('b',self.pool([1,2,100,200]))
  self.assertEqual(len(x),100);self.assertEqual(a['test:0:0']['selected_n'],1);self.assertEqual(a['test:1:0']['selected_n'],2)
 def test_census(self):
  rows=self.pool([10,10]);x,a=select('b',rows,20)
  self.assertTrue(all(r['conditional_inclusion_probability']==1 for r in x.values()))
 def test_distinct_backbone_keys(self):
  rows=self.pool([25]*20)
  self.assertNotEqual(set(select('a',rows)[0]),set(select('b',rows)[0]))
 def test_100_is_nested_in_110(self):
  for counts in [[25]*20,[1,2,5,6,7,10,50,80,100,239]]:
   rows=self.pool(counts)
   self.assertTrue(set(select('a',rows,100)[0])<=set(select('a',rows,110)[0]))
 def test_reject_mixed_split(self):
  rows=self.pool([500]);rows[0]['split']='val'
  with self.assertRaises(AssertionError):select('b',rows)
if __name__=='__main__':unittest.main()
