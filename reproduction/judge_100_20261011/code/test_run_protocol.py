"""CPU-only validation of the judge output contract and durable artifact helpers."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import run_judge
from common import read, write, sha


class OutputContractTests(unittest.TestCase):
    def setUp(self):
        self.prompt = dict(sample_id='dataset:example', prompt_id='hard:example:nearest', dataset='dataset', backbone='backbone', condition='nearest', token_ids=[7, 8, 9], input_tokens=3, instruction_sha256='fixed-hash')

    def output(self, text='1', token_ids=None, finish_reason='length', input_ids=None):
        g=SimpleNamespace(text=text, token_ids=[16] if token_ids is None else token_ids, finish_reason=finish_reason)
        return SimpleNamespace(prompt_token_ids=self.prompt['token_ids'] if input_ids is None else input_ids, outputs=[g])

    def records(self, output, mode='hard'):
        return run_judge.make_records([self.prompt], [output], mode, 4, .75, [15,16])

    def test_hard_records_preserve_identity_and_cost(self):
        r=self.records(self.output())[0]
        self.assertEqual(r['prediction'],1)
        self.assertEqual(r['prompt_id'], self.prompt['prompt_id'])
        self.assertEqual(r['sample_id'], self.prompt['sample_id'])
        self.assertEqual(r['input_tokens'],3)
        self.assertEqual(r['batch_number'],4)
        self.assertEqual(r['batch_seconds'],.75)
        self.assertEqual(r['generated_text'],'1')
        self.assertEqual(r['token_ids'],[16])

    def test_both_hard_labels_and_mistral_token_ids(self):
        for label,tid in [(0,1048),(1,1049)]:
            r=run_judge.make_records([self.prompt],[self.output(str(label),[tid])],'hard',0,1,[1048,1049])[0]
            self.assertEqual(r['prediction'],label)

    def test_probability_boundaries_and_downstream_threshold(self):
        for n in [0,1,49,50,99,100]:
            r=self.records(self.output(str(n),[101,102],'stop'),'probability')[0]
            self.assertEqual(r['positive_probability'],n)
            self.assertEqual(r['prediction_at_50'],int(n>=50))
            self.assertNotIn('prediction',r)

    def test_preserves_raw_whitespace_while_parsing_integer(self):
        r=self.records(self.output(' 50\n',[50],'stop'),'probability')[0]
        self.assertEqual(r['positive_probability'],50)
        self.assertEqual(r['generated_text'],' 50\n')

    def test_rejects_noncontract_probability_outputs(self):
        for text in ['', '-1','101','100.0','0.5','50%','01','+50','{"probability":50}','positive','50 because ...','5\n0']:
            with self.subTest(text=text), self.assertRaises(AssertionError):
                self.records(self.output(text,[123],'stop'),'probability')

    def test_rejects_noncontract_hard_outputs(self):
        for text in ['', '2','10','yes','1.0','label: 1','1 because ...']:
            with self.subTest(text=text), self.assertRaises(AssertionError):
                self.records(self.output(text))

    def test_rejects_truncated_probability(self):
        with self.assertRaises(AssertionError):
            self.records(self.output('50',[50],'length'),'probability')

    def test_rejects_wrong_input_tokens(self):
        with self.assertRaises(AssertionError):
            self.records(self.output(input_ids=[7,8,10]))

    def test_rejects_multiple_generations(self):
        o=self.output();o.outputs*=2
        with self.assertRaises(AssertionError):self.records(o)

    def test_rejects_missing_output(self):
        with self.assertRaises(AssertionError):
            run_judge.make_records([self.prompt],[],'hard',0,0,[15,16])

    def test_rejects_wrong_hard_label_token(self):
        with self.assertRaises(AssertionError):self.records(self.output(token_ids=[99]))

    def test_rejects_multiple_hard_tokens(self):
        with self.assertRaises(AssertionError):self.records(self.output(token_ids=[16,16]))

    def test_hard_text_must_agree_with_token(self):
        with self.assertRaises(AssertionError):self.records(self.output('0',[16]))

    def test_hard_finish_reason_must_be_normal(self):
        with self.assertRaises(AssertionError):self.records(self.output(finish_reason='error'))


class ResumeTests(unittest.TestCase):
    setUp=OutputContractTests.setUp
    output=OutputContractTests.output
    def batch(self, mode='hard'):
        p2=dict(self.prompt, sample_id='dataset:other', prompt_id='hard:other:nearest', token_ids=[10,11])
        chunk=[self.prompt,p2]
        outputs=[self.output(),self.output('0',[15],input_ids=[10,11])]
        if mode=='probability':
            for o in outputs:o.outputs[0].finish_reason='stop'
        records=run_judge.make_records(chunk,outputs,mode,4,.75,[15,16])
        raw=[dict(prompt_token_ids=o.prompt_token_ids,outputs=[vars(g) for g in o.outputs]) for o in outputs]
        return chunk,dict(records=records),raw

    def verify(self,chunk,doc,raw,mode='hard'):
        return run_judge.validate_committed_batch(chunk,doc,raw,mode,4,[15,16])

    def test_exact_resume_rebuilds_same_records(self):
        for mode in ['hard','probability']:
            chunk,doc,raw=self.batch(mode)
            self.assertEqual(self.verify(chunk,doc,raw,mode),doc['records'])

    def test_resume_rejects_tampered_record_identity_and_label(self):
        for field,value in [('sample_id','wrong'),('prompt_id','wrong'),('prediction',0),('dataset','wrong'),('backbone','wrong'),('mode','probability'),('batch_number',100),('instruction_sha256','changed')]:
            chunk,doc,raw=self.batch();doc['records'][0][field]=value
            with self.subTest(field=field),self.assertRaises(AssertionError):self.verify(chunk,doc,raw)

    def test_resume_rejects_reordered_raw_outputs(self):
        chunk,doc,raw=self.batch()
        with self.assertRaises(AssertionError):self.verify(chunk,doc,list(reversed(raw)))

    def test_resume_rejects_changed_preserved_generation(self):
        chunk,doc,raw=self.batch();raw[0]['outputs'][0].update(text='0',token_ids=[15])
        with self.assertRaises(AssertionError):self.verify(chunk,doc,raw)

    def test_resume_rejects_partial_outputs(self):
        chunk,doc,raw=self.batch()
        with self.assertRaises(AssertionError):self.verify(chunk,doc,raw[:1])

    def test_resume_rejects_inconsistent_batch_timing(self):
        chunk,doc,raw=self.batch();doc['records'][1]['batch_seconds']=9
        with self.assertRaises(AssertionError):self.verify(chunk,doc,raw)

    def test_resume_rejects_truncated_probability_raw(self):
        chunk,doc,raw=self.batch('probability');raw[0]['outputs'][0]['finish_reason']='length'
        with self.assertRaises(AssertionError):self.verify(chunk,doc,raw,'probability')


class ArtifactTests(unittest.TestCase):
    def test_atomic_write_roundtrip_and_checksum_changes(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'nested'/'artifact.json';write(p,{'state':'running','n':16})
            initial=sha(p);self.assertEqual(read(p),{'state':'running','n':16})
            self.assertFalse(p.with_name(p.name+'.tmp').exists())
            write(p,{'state':'completed','n':32});self.assertNotEqual(sha(p),initial)
            self.assertEqual(read(p)['state'],'completed')

    def test_nonfinite_write_cannot_replace_committed_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'artifact.json';write(p,{'n':16});initial=sha(p)
            with self.assertRaises(ValueError):write(p,{'n':float('nan')})
            self.assertEqual(sha(p),initial)
            self.assertEqual(read(p),{'n':16})

    def test_specs_native_template_contracts(self):
        p=Path(__file__).parents[1]/'judge_specs.json'
        d=json.loads(p.read_text())
        for key,s in d['models'].items():
            self.assertEqual(s['engine']['max_num_seqs'],s['batch_size'])
            self.assertNotIn('model',s['engine'])
            self.assertNotIn('max_model_len',s['engine'])
            if key=='mistral':
                self.assertTrue(s['chat_template_kwargs']['tokenize'])
                self.assertEqual(s['chat_template_kwargs']['reasoning_effort'],'none')
                self.assertIn('transformers_5_4_overlay',s['environment']['PYTHONPATH'])
            else:self.assertFalse(s['chat_template_kwargs']['enable_thinking'])

if __name__=='__main__':unittest.main(verbosity=2)
