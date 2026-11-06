import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from models.train_xgboost import report_parameters


class XGBoostOutputTests(unittest.TestCase):
    def test_missing_parameter_encoding(self):
        model = xgb.XGBClassifier()
        parameters = report_parameters(model)
        self.assertIsNone(json.loads(json.dumps(parameters, allow_nan=False))['missing'])
        self.assertTrue(np.isnan(model.get_params()['missing']))
        self.assertEqual(report_parameters(xgb.XGBClassifier(missing=-999))['missing'], -999)

    def test_training_writes_strict_json_and_reloadable_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame([
                {'split': split, 'label': i % 2,
                 'p1_card_1_level': float(i % 2),
                 'p2_card_1_level': float(1 - i % 2)}
                for split in ('train', 'validation', 'test') for i in range(20)
            ])
            source = root / 'baseline.csv'
            frame.to_csv(source, index=False)
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / 'models/train_xgboost.py'),
                '--input', str(source), '--output-dir', str(root / 'output'),
                '--n-estimators', '2', '--threads', '1',
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / 'output/deck-only'
            def reject_constant(value):
                raise ValueError(value)
            for name in ('metrics', 'features', 'learning_curve'):
                json.loads((output / f'{name}.json').read_text(), parse_constant=reject_constant)
            model = xgb.XGBClassifier()
            model.load_model(output / 'model.json')
            predictions = model.predict_proba(frame[['p1_card_1_level', 'p2_card_1_level']])
            self.assertTrue(np.isfinite(predictions).all())
