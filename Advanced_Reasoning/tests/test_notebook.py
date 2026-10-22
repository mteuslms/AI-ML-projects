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
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

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

if __name__ == '__main__':
    unittest.main()
