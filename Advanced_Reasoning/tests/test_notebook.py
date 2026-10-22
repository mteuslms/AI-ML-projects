"""Offline regression checks: python -m unittest discover -s Advanced_Reasoning/tests -v"""
import ast
import json
from pathlib import Path
import tempfile
import unittest

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Socratic_Tutor.ipynb'

class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())
        cls.scope = {'Path': Path, 'json': json}
        for cell in cls.notebook['cells']:
            if 'helpers' in cell.get('metadata', {}).get('tags', []):
                exec(''.join(cell['source']), cls.scope)

    def test_all_python_cells_compile(self):
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if not source.startswith('%pip'):
                    ast.parse(source)
                self.assertIsInstance(cell['outputs'], list)
                self.assertFalse(any(o.get('output_type') == 'error' for o in cell['outputs']))
                count = cell['execution_count']
                self.assertTrue(count is None or type(count) is int)

    def test_missing_scores_are_not_zero(self):
        result = self.scope['summarize']([{'score':2}, {'score':None}, {'score':0}])
        self.assertEqual(result['normalized_score_pct'], 50)
        self.assertEqual(result['strict_sar_pct'], 50)
        self.assertEqual(result['missing'], 1)
        self.assertAlmostEqual(result['coverage_pct'], 200/3)

    def test_all_missing_and_empty(self):
        self.assertIsNone(self.scope['summarize']([{'score':None}])['strict_sar_pct'])
        self.assertIsNone(self.scope['summarize']([])['coverage_pct'])

    def test_metrics_distinguish_partial_credit(self):
        summary = self.scope['summarize']([{'score':1}, {'score':2}])
        self.assertEqual(summary['normalized_score_pct'], 75)
        self.assertEqual(summary['strict_sar_pct'], 50)

    def test_judge_rejects_bad_scores(self):
        for value in (True, '2', 2.5, -1, 3, None):
            with self.assertRaises(ValueError):
                self.scope['parse_score'](json.dumps({'score':value}))
        self.assertEqual(self.scope['parse_score']('{"score":2}'), 2)

    def test_dataset_validation_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            rows = [{'prompt':f'Question {i}', 'chosen':'What have you tried?', 'rejected':'Answer.'} for i in range(10)]
            rows.append(dict(rows[0], prompt='  QUESTION   0  '))
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(len(self.scope['load_rows'](path)), 10)
            path.write_text('{"prompt": "missing responses"}\n')
            with self.assertRaisesRegex(ValueError, 'line 1'):
                self.scope['load_rows'](path)

    def test_invalid_preferences(self):
        for row in ({}, {'prompt':'x','chosen':'y','rejected':'y'}, {'prompt':'x','chosen':'','rejected':'z'}):
            with self.assertRaises(ValueError):
                self.scope['validate_example'](row)

    def test_attack_inventory(self):
        scope = {}
        for cell in self.notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type']=='code' and 'adversarial_prompts_full = {' in source:
                exec(source, scope)
        from collections import Counter
        prompts = scope['adversarial_prompts_full']
        self.assertEqual(Counter(k.split('_')[1] for k in prompts),
                         {'Emotional':10,'Authority':10,'Context':10,'Stealth':5})


    def test_public_defaults_skip_training_and_paid_calls(self):
        scope = {}
        with tempfile.TemporaryDirectory() as directory:
            import os
            previous = os.getcwd()
            try:
                os.chdir(directory)
                for cell in self.notebook['cells']:
                    source = ''.join(cell['source'])
                    if cell['cell_type'] == 'code' and 'def api_client():' in source:
                        exec(source, scope)
            finally:
                os.chdir(previous)
        self.assertEqual(scope['MODE'], 'inspect')
        self.assertIs(scope['GENERATE_DATA'], False)
        self.assertIs(scope['RUN_API_JUDGE'], False)

    def test_recorded_summary_is_consistent(self):
        import csv
        path = NOTEBOOK.parent / 'results' / 'recorded_summary.csv'
        with path.open() as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(int(record['scored']), 35)
            self.assertEqual(int(record['missing']), 0)
            self.assertAlmostEqual(float(record['strict_sar_pct']),
                                   100*int(record['strict_successes'])/35)
        self.assertEqual([int(r['strict_successes']) for r in records], [31, 34, 2, 2])

    def test_legacy_colab_secret_fallback(self):
        import os
        import sys
        import types
        from unittest.mock import patch
        scope = {}
        sdk = types.ModuleType('openai')
        sdk.OpenAI = lambda api_key: api_key
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        def get_secret(name):
            if name == 'OPENAI_API_KEY':
                raise KeyError(name)
            return 'dummy-test-value'
        colab.userdata = types.SimpleNamespace(get=get_secret)
        google.colab = colab
        modules = {'openai': sdk, 'google': google, 'google.colab': colab}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True):
            previous = os.getcwd()
            try:
                os.chdir(directory)
                for cell in self.notebook['cells']:
                    source = ''.join(cell['source'])
                    if cell['cell_type'] == 'code' and 'def api_client():' in source:
                        exec(source, scope)
                self.assertEqual(scope['api_client'](), 'dummy-test-value')
            finally:
                os.chdir(previous)

    def test_judging_resume_prefers_saved_scores(self):
        import random
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            saved = {'condition': [{'id': 'test', 'score': 2}]}
            raw = {'condition': [{'id': 'test', 'score': None}]}
            (path / 'eval_results.json').write_text(json.dumps(saved))
            (path / 'eval_responses.json').write_text(json.dumps(raw))
            scope = {'RUN_API_JUDGE': True, 'results': {}, 'WORK_DIR': path,
                     'json': json, 'random': random, 'SEED': 42,
                     'api_client': lambda: object(),
                     'save_json': lambda *args: self.fail('No new judge results expected')}
            for cell in self.notebook['cells']:
                source = ''.join(cell['source'])
                if cell['cell_type'] == 'code' and 'def judge_reply(' in source:
                    exec(source, scope)
            self.assertEqual(scope['results'], saved)
            self.assertEqual(scope['pending'], [])

if __name__ == '__main__':
    unittest.main()

