import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from models.predict_independent_deck import evaluate, predict, select_rows
from models.train_independent_deck import train
from tests.test_independent_deck import match, tiny_dataset, write_matches

ROOT = Path(__file__).resolve().parents[1]


def train_tiny(root, rows, feature_mode='deck-only'):
    source = root / 'matches.jsonl'
    write_matches(source, rows)
    train(Namespace(
        input=source, output_dir=root / 'artifacts', features=feature_mode, evaluate_test=False,
        embed_dim=8, encoder_hidden=8, encoder_out=8, mlp_hidden=8, dropout=0.0,
        epochs=1, patience=1, batch_size=8, learning_rate=1e-3, weight_decay=0.0,
        threads=1, seed=0,
    ))
    return source, root / 'artifacts' / feature_mode


class IndependentDeckInferenceTests(unittest.TestCase):
    def test_checkpoint_prediction_ignores_card_order_and_unknown_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = tiny_dataset()
            _, model_dir = train_tiny(root, rows)
            validation = [row for row in rows if row['split'] == 'validation']
            original = predict(model_dir, validation)
            self.assertEqual(len(original), len(validation))
            self.assertTrue(np.isfinite(original['p1_win_probability']).all())
            self.assertTrue(set(original['predicted_p1_win']).issubset({0, 1}))

            shuffled = []
            for row in validation:
                clone = copy.deepcopy(row)
                for side in ('p1', 'p2'):
                    clone[side]['cards'] = list(reversed(clone[side]['cards']))
                shuffled.append(clone)
            reordered = predict(model_dir, shuffled)
            np.testing.assert_allclose(
                original['p1_win_probability'], reordered['p1_win_probability'], atol=1e-5,
            )

            unknown = copy.deepcopy(validation[:1])
            unknown[0]['p1']['cards'][0]['id'] = 999999
            for side in ('p1', 'p2'):
                del unknown[0][side]['starting_trophies']
            scored = predict(model_dir, unknown)
            self.assertTrue(np.isfinite(scored['p1_win_probability']).all())
            self.assertEqual(scored['match_id'].tolist(), [unknown[0]['match_id']])

    def test_cli_reports_validation_and_keeps_previous_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, model_dir = train_tiny(root, tiny_dataset(), 'with-player-stats')
            command = [
                sys.executable, str(ROOT / 'models' / 'predict_independent_deck.py'),
                '--model-dir', str(model_dir), '--input', str(source),
                '--limit', '1', '--output', str(root / 'out.csv'), '--batch-size', '4',
            ]
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env)
            self.assertEqual(run.returncode, 0, (run.stdout or '') + (run.stderr or ''))
            self.assertIn('Split: validation', run.stderr)
            self.assertIn('Matches: 1', run.stderr)
            self.assertIn('N/A', run.stderr)
            self.assertEqual(run.stdout, '')
            saved = pd.read_csv(root / 'out.csv')
            self.assertEqual(list(saved['match_id']), ['validation-0'])
            self.assertEqual(subprocess.run(command, capture_output=True, text=True, encoding='utf-8', env=env).returncode, 0)
            self.assertTrue((root / 'out-1.csv').exists())

            unlabeled = match('validation', 1, list(range(1, 9)), list(range(9, 17)))
            del unlabeled['label']
            unlabeled_path = root / 'unlabeled.jsonl'
            unlabeled_path.write_text(json.dumps(unlabeled) + '\n', encoding='utf-8')
            quiet = subprocess.run(command + [
                '--input', str(unlabeled_path), '--split', 'all', '--no-evaluate',
                '--output', str(root / 'plain.csv'),
            ], capture_output=True, text=True, encoding='utf-8', env=env)
            self.assertEqual(quiet.returncode, 0, quiet.stderr)
            self.assertNotIn('Accuracy', quiet.stderr)
            self.assertEqual(len(pd.read_csv(root / 'plain.csv')), 1)

            missing = subprocess.run([
                sys.executable, str(ROOT / 'models' / 'predict_independent_deck.py'),
                '--model-dir', str(root / 'missing'), '--input', str(source),
                '--output', str(root / 'unused.csv'),
            ], capture_output=True, text=True, encoding='utf-8', env=env)
            self.assertEqual(missing.returncode, 2)
            self.assertIn('config.json', missing.stderr)

    def test_split_filter_and_single_class_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'matches.jsonl'
            write_matches(path, tiny_dataset())
            selected = select_rows(path, 'test', 2)
            self.assertEqual([row['split'] for row in selected], ['test', 'test'])
            nosplit = Path(tmp) / 'nosplit.jsonl'
            nosplit.write_text(json.dumps({'label': 1}) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'split field'):
                select_rows(nosplit, 'validation', None)
            report = evaluate(
                pd.DataFrame({'label': [1]}),
                pd.DataFrame({'p1_win_probability': [0.8], 'predicted_p1_win': [1]}),
            )
            self.assertIsNone(report['roc_auc'])
            self.assertEqual(report['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
