"""Focused checks for numerical provenance, weights, ties, and token costs."""
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

import evaluate_expansion as adapter


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.hard = self.root / 'hard'
        self.prob = self.root / 'probability'
        self.out = self.root / 'analysis'
        self.out.mkdir()
        self.ns = adapter.load_stage('probability_evaluate.py', self.hard, self.prob, self.out)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sources_frozen_and_functions_identical(self):
        for name in adapter.read(adapter.HERE / 'UPSTREAM_SHA256.json'):
            adapter.source(name)
        self.assertEqual(self.ns['ROOT'], self.prob)
        self.assertEqual(self.ns['PARENT'], self.hard)
        self.assertEqual(self.ns['JS'], adapter.JS)
        self.assertEqual(adapter.PAIRS, 350)
        self.assertEqual(adapter.PER_MODE, 1050000)
        # Every estimator syntax tree is unchanged; only outer routing changes.
        original = ast.parse(adapter.source('probability_evaluate.py'))
        math_functions = [x for x in original.body if isinstance(x, ast.FunctionDef) and x.name != 'main']
        adapted = {x.name:x for x in ast.parse(self.ns['__adapted_source__']).body if isinstance(x, ast.FunctionDef)}
        for f in math_functions:
            self.assertEqual(ast.dump(f), ast.dump(adapted[f.name]), f.name)

    def test_joint_surface_matches_brute_force(self):
        rows = [
            dict(probe_probability=.1, true_label=0, inclusion_probability=.5),
            dict(probe_probability=.4, true_label=1, inclusion_probability=1.),
            dict(probe_probability=.8, true_label=0, inclusion_probability=.25),
            dict(probe_probability=.9, true_label=1, inclusion_probability=.25),
        ]
        scores = np.array([.1, .4, .8, .9])
        probabilities = np.array([10, 70, 55, 90])
        weights = np.array([2, 1, 4, 4])
        gold = np.array([0, 1, 0, 1])
        thresholds = np.array([0., .1, .4, .6, .8, .9, 1.])
        pop = dict(n=11, positive=5)
        g = self.ns['grid'](rows, probabilities, thresholds, pop)
        for cutoff in range(102):
            for k, threshold in enumerate(thresholds):
                yes = (scores >= threshold) & (probabilities >= cutoff)
                tp = np.sum(weights * gold * yes)
                fp = np.sum(weights * (1-gold) * yes)
                self.assertEqual(g['tp'][cutoff,k], tp)
                self.assertEqual(g['fp'][cutoff,k], fp)
                self.assertAlmostEqual(g['f1'][cutoff,k], 2*tp/(5+tp+fp))

    def test_ties_choose_highest_gate_then_highest_cutoff(self):
        grid = np.ones((102, 3))
        self.assertEqual(self.ns['choose'](grid, np.array([0., .5, 1.])), (101, 2))
        self.assertEqual(self.ns['choose'](grid, np.array([0., .5, 1.]), 50), (50, 2))

    def test_stratified_resamples_keep_census_fixed(self):
        rows = [dict(stratum='census', inclusion_probability=1.),
                dict(stratum='sampled', inclusion_probability=.5),
                dict(stratum='sampled', inclusion_probability=.5)]
        m = self.ns['bootstrap'](rows, np.random.default_rng(1), reps=20)
        np.testing.assert_array_equal(m[:,0], np.ones(20))
        np.testing.assert_array_equal(m[:,1:].sum(axis=1), np.full(20,2))
        np.testing.assert_array_equal(m, self.ns['bootstrap'](rows, np.random.default_rng(1), reps=20))

    def test_hard_and_probability_counts_agree_for_binary_extremes(self):
        hard = adapter.load_stage('hard_evaluate.py', self.hard, self.prob, self.out)
        rows = [dict(probe_probability=.2,true_label=0,inclusion_probability=.5),
                dict(probe_probability=.7,true_label=1,inclusion_probability=.25)]
        pred = np.array([0,1])
        ts = np.array([0.,.3,.9]); pop = dict(n=6,positive=4)
        h = hard['curves'](rows,pred,ts,pop)
        p = self.ns['curve'](rows,pred,ts,pop)
        for field in ['tp','fp','f1']:
            np.testing.assert_array_equal(h[field],p[field])

    def test_comprehensive_analyzer_routing_and_numerical_functions(self):
        # The comprehensive analyzer imports pyplot but never invokes it. Stub
        # that optional import for local environments with NumPy but no plots.
        mpl = types.ModuleType('matplotlib'); mpl.use = lambda *_: None
        plt = types.ModuleType('matplotlib.pyplot'); mpl.pyplot = plt
        with mock.patch.dict('sys.modules', {'matplotlib':mpl, 'matplotlib.pyplot':plt}):
            ns = adapter.load_stage('analyze.py',self.hard,self.prob,self.out)
        self.assertEqual((ns['P'],ns['R'],ns['OUT']), (self.hard,self.prob,self.out))
        self.assertEqual(ns['JS'],adapter.JS)
        old = {x.name:x for x in ast.parse(adapter.source('analyze.py')).body if isinstance(x,ast.FunctionDef)}
        new = {x.name:x for x in ast.parse(ns['__adapted_source__']).body if isinstance(x,ast.FunctionDef)}
        for name in ['cm','f1','resamples','prepare_grid','surface','hardcurve','select','winner','metrics_yes','boot_f1']:
            self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)

    def test_token_cost_depends_on_gate_not_judge_decision(self):
        rows = [dict(sample_id='a',dataset='tiny',split='test',probe_probability=.2,inclusion_probability=.2),
                dict(sample_id='b',dataset='tiny',split='test',probe_probability=.8,inclusion_probability=.1)]
        adapter.write(self.prob/'bb/selected_flags.json',rows)
        variants={
            'probe':dict(routing_rate=0.,f1=.5,precision=.6,recall=.7),
            'baseline_parent_hard':dict(probe_threshold=.5,routing_rate=.1,f1=.5,precision=.6,recall=.7),
            'baseline_joint':dict(probe_threshold=.1,probability_threshold=101,routing_rate=.2,f1=0.,precision=0.,recall=0.),
        }
        pair=dict(backbone='bb',judge='j',dataset='tiny',populations=dict(test=dict(n=100)),variants=variants)
        adapter.write(self.out/'pairs.json',[pair])
        for root in [self.hard,self.prob]:
            for condition in adapter.CS:
                p=root/'bb/judges'/f'j_{condition}'/'run/judgments.jsonl';p.parent.mkdir(parents=True)
                p.write_text('\n'.join(json.dumps(dict(sample_id=s,input_tokens=t)) for s,t in [('a',10),('b',20)])+'\n')
        with mock.patch.object(adapter,'BBS',['bb']),mock.patch.object(adapter,'JS',['j']):
            adapter.token_costs(self.hard,self.prob,self.out)
        costs={r['variant']:r for r in adapter.read(self.out/'input_token_costs.json')['rows']}
        self.assertEqual(costs['probe']['estimated_population_input_tokens'],0.)
        self.assertEqual(costs['baseline_parent_hard']['estimated_population_input_tokens'],200.)
        self.assertEqual(costs['baseline_joint']['estimated_population_input_tokens'],250.)
        self.assertEqual(costs['baseline_joint']['estimated_input_tokens_per_1000_population_examples'],2500.)


if __name__ == '__main__':
    unittest.main()
