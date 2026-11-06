import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from models.predict_xgboost import predict


class InferenceTests(unittest.TestCase):
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
                       '--split', 'validation', '--limit', '1', '--output', str(root / 'out.csv')]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            output = pd.read_csv(root / 'out.csv')
            self.assertEqual(output['match_id'].tolist(), ['b'])
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
