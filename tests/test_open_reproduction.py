import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from open_reproduction.common import validate_rows
from open_reproduction.judge_worker import dispatch, flip_cases, save_results
from open_reproduction.probes import fit_probe, select, top_two
from open_reproduction.prompts import classification, mechanism, reasoning, TEXT
from paper.collect import initialise


def data():
    rows = []
    x = []
    for s in ['train','val','test']:
        for i in range(20):
            uid = f'{s}:{i}'
            rows.append(dict(immutable_id=uid,label=i%2,split=s,group=uid,input_hash=uid))
            x.append([i%2*2-1,i/20])
    return rows,np.array(x,dtype=np.float32)


class RebuildTests(unittest.TestCase):
    def test_scaler_and_fit_use_training_only(self):
        rows,x = data()
        a,p = fit_probe(rows,x,'toxicchat')
        x2 = x.copy()
        x2[40:] += 300
        b,q = fit_probe(rows,x2,'toxicchat')
        np.testing.assert_array_equal(a['scaler'].mean_,b['scaler'].mean_)
        np.testing.assert_array_equal(a['model'].coef_,b['model'].coef_)
        self.assertEqual(a['threshold'],b['threshold'])

    def test_no_group_or_text_leakage(self):
        rows,_ = data()
        rows[-1]['group'] = rows[0]['group']
        with self.assertRaisesRegex(ValueError,'leakage'):
            validate_rows(rows)

    def test_retrieval_distinct_ties_across_chunks(self):
        train = np.array([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]])
        ids,scores = top_two(np.array([[1.,0.],[0.,1.]]),train,chunk=1)
        self.assertEqual(ids.tolist(),[[0,1],[2,0]])
        np.testing.assert_array_equal(scores,[[1,1],[1,0]])

    def test_sampling_reproducible_with_population_weights(self):
        rows,_ = data()
        p = np.tile([.2,.8],len(rows)//2)
        selected = select(rows,p,'llama','toxicchat',12)
        self.assertEqual(selected,select(rows,p,'llama','toxicchat',12))
        for s in ['val','test']:
            self.assertEqual(sum(r['split']==s for r in selected),12)
            self.assertAlmostEqual(sum(r['population_weight'] for r in selected if r['split']==s),20)

    def test_prompt_contract_and_private_fields(self):
        row = dict(dataset='toxicchat',messages=[dict(role='user',content='target')],saliency={'sentence':'target'},nearest=dict(messages=[dict(role='user',content='reference')],label=1,prediction=0),probe_probability=.61,true_label='PRIVATE_GOLD',original_score='PRIVATE_SCORE',outcome='PRIVATE_OUTCOME')
        nn = classification(row,'nearest')
        self.assertTrue(nn[1]['content'].endswith('Positive-class probability (0-100):'))
        self.assertIn('Dataset label: toxic (1)',nn[1]['content'])
        self.assertIn('Probe prediction: non-toxic (0)',nn[1]['content'])
        self.assertNotIn('PRIVATE_',json.dumps(nn))
        after = reasoning(nn,'reason_after')
        self.assertEqual(nn[1],after[1])
        self.assertIn('VERY briefly',after[0]['content'])
        first = reasoning(nn,'reason_first')
        self.assertTrue(first[1]['content'].endswith('Assessment (VERY brief reasoning, then positive-class probability):'))
        self.assertNotIn('PRIVATE_',json.dumps(mechanism(row)))

    def test_dispatch_and_raw_recovery_without_regeneration(self):
        with tempfile.TemporaryDirectory() as d:
            con = initialise(Path(d)/'responses.sqlite',{})
            job = dict(prompt_id='x',study='reason_after',case_key='case',condition='reason_after',split='test',messages=[dict(role='user',content='hi')])
            dispatch(con,'judge',[job])
            with self.assertRaises(sqlite3.IntegrityError):
                dispatch(con,'judge',[job])
            raw = dict(prompt_id='x',study='reason_after',text='73\nCut short',finish_reason='length',time=1,model='pinned',seconds=.2,input_token_ids=[1],output_token_ids=[2,3])
            save_results(con,'judge',[raw])
            self.assertEqual(con.execute('SELECT parsed,status,finish_reason FROM requests').fetchone(),('73','valid','length'))
            self.assertIn('Cut short',con.execute('SELECT raw_json FROM requests').fetchone()[0])

    def test_reasoning_flips_are_judge_specific_and_test_only(self):
        with tempfile.TemporaryDirectory() as d:
            con = initialise(Path(d)/'r.sqlite',{})
            for judge,case,split,a,b in [('a','flip','test',20,80),('b','stable','test',60,80),('a','validation','val',20,80)]:
                for cond,score in [('baseline',a),('nearest',b)]:
                    con.execute('INSERT INTO requests(model,prompt_id,study,case_key,condition,split,status,parsed) VALUES (?,?,?,?,?,?,?,?)',(judge,case+cond,'main',case,cond,split,'valid',str(score)))
            con.commit()
            self.assertEqual(flip_cases(con,'a'),{'flip'})
            self.assertEqual(flip_cases(con,'b'),set())


if __name__ == '__main__':
    unittest.main()
