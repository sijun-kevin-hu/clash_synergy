import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from rich.console import Console
from models.predict_xgboost import predict, evaluate, show_report, save_predictions


class InferenceTests(unittest.TestCase):
    def test_output_numbers_increase_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'batch.csv'
            frame = pd.DataFrame({'prediction': [1]})
            self.assertEqual(save_predictions(frame, path), path)
            original = path.read_bytes()
            self.assertEqual(save_predictions(frame, path).name, 'batch-1.csv')
            self.assertEqual(save_predictions(frame, path).name, 'batch-2.csv')
            (path.parent / 'batch-9.csv').write_text('existing')
            self.assertEqual(save_predictions(frame, path).name, 'batch-10.csv')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual((path.parent / 'batch-9.csv').read_text(), 'existing')

    def test_metrics_use_row_order_and_thresholded_predictions(self):
        rows = pd.DataFrame({'label': [0, 1, 1, 0]}, index=[8, 12, 16, 20])
        predictions = pd.DataFrame({'p1_win_probability': [.1, .8, .6, .4],
                                    'predicted_p1_win': [0, 1, 0, 0]})
        report = evaluate(rows, predictions)
        self.assertEqual(report['accuracy'], .75)
        self.assertAlmostEqual(report['classification'][1]['precision'], 1.)
        self.assertAlmostEqual(report['classification'][1]['recall'], .5)
        self.assertAlmostEqual(report['classification'][1]['f1'], 2 / 3)
        stream = io.StringIO()
        show_report(Console(file=stream, width=100, color_system=None), report, 'model', 'validation', .7)
        rendered = stream.getvalue()
        for label in ('Macro F1', 'Precision', 'Recall', 'Confusion matrix', '75.00%'):
            self.assertIn(label, rendered)
        self.assertNotIn('\x1b', rendered)
        self.assertEqual(report['roc_auc'], 1.)
        self.assertEqual(report['confusion_matrix'], [[2, 0], [1, 1]])
        self.assertAlmostEqual(report['brier_score'], .0925)
        self.assertAlmostEqual(report['log_loss'], -np.log([.9, .8, .6, .6]).mean())
        self.assertIsNone(evaluate(rows.iloc[:1], predictions.iloc[:1])['roc_auc'])
        for invalid in (pd.DataFrame({'label': [2]}), pd.DataFrame({'other': [0]})):
            with self.assertRaisesRegex(ValueError, 'label column'):
                evaluate(invalid, predictions.iloc[:1])

    def test_cli_and_feature_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            X = pd.DataFrame({'a': [0., 1., 0., 1.], 'b': [1., 0., 1., 0.]})
            model = XGBClassifier(n_estimators=2, n_jobs=1)
            model.fit(X, [0, 1, 0, 1])
            model.save_model(root / 'model.json')
            (root / 'features.json').write_text(json.dumps(['a', 'b']))
            rows = X[['b', 'a']].assign(match_id=['a', 'b', 'c', 'd'],
                                         split=['train', 'validation', 'validation', 'test'], label=1)
            np.testing.assert_allclose(predict(root, rows)['p1_win_probability'], model.predict_proba(X)[:, 1])
            with self.assertRaisesRegex(ValueError, 'Missing required'):
                predict(root, rows.drop(columns='a'))
            rows.to_csv(root / 'input.csv', index=False)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'models/predict_xgboost.py'),
                       '--model-dir', str(root), '--input', str(root / 'input.csv'),
                       '--split', 'validation', '--limit', '1', '--evaluate', '--output', str(root / 'out.csv')]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('N/A', run.stderr)
            self.assertIn('Matches: 1', run.stderr)
            output = pd.read_csv(root / 'out.csv')
            self.assertEqual(output['match_id'].tolist(), ['b'])
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertTrue((root / 'out-1.csv').exists())
