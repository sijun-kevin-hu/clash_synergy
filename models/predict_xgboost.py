"""Run saved XGBoost inference on CSV rows encoded by preprocessing."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss, confusion_matrix


def evaluate(rows, predictions):
    if 'label' not in rows or not rows['label'].isin([0, 1]).all():
        raise ValueError('--evaluate requires a label column containing only 0 or 1.')
    y = rows['label'].to_numpy()
    p = predictions['p1_win_probability'].to_numpy()
    predicted = predictions['predicted_p1_win'].to_numpy()
    return {
        'matches': len(y),
        'accuracy': float(accuracy_score(y, predicted)),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y, p)),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def predict(model_dir, rows, threshold=0.5):
    features = json.loads((model_dir / 'features.json').read_text())
    if not isinstance(features, list) or not features or not all(isinstance(c, str) for c in features):
        raise ValueError('features.json must contain a nonempty list of column names.')
    missing = set(features) - set(rows.columns)
    if missing:
        raise ValueError(f'Missing required feature columns: {sorted(missing)}')
    X = rows[features].astype('float32')
    if np.isinf(X.to_numpy()).any():
        raise ValueError('Features contain infinity; missing values may be NaN.')
    model = XGBClassifier()
    model.load_model(model_dir / 'model.json')
    probabilities = model.predict_proba(X)[:, 1]
    result = pd.DataFrame({
        'p1_win_probability': probabilities,
        'predicted_p1_win': (probabilities >= threshold).astype('int8'),
    })
    if 'match_id' in rows.columns:
        result.insert(0, 'match_id', rows['match_id'].to_numpy())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/xgboost/deck-only'))
    parser.add_argument('--input', type=Path, default=Path('data/processed/combined/baseline.csv'))
    parser.add_argument('--output', type=Path, help='Save predictions as CSV; otherwise print CSV to stdout.')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], help='Select rows by their existing split.')
    parser.add_argument('--limit', type=int, help='Predict only the first N rows after split selection.')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--evaluate', action='store_true', help='Print labeled batch metrics to stderr; ROC-AUC is null for a single-class batch.')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    try:
        if args.output and args.output.exists():
            raise ValueError(f'Output already exists: {args.output}; choose a new path.')
        rows = pd.read_csv(args.input, nrows=args.limit if not args.split else None)
        if args.split:
            if 'split' not in rows.columns:
                raise ValueError('--split requires a split column in the input CSV.')
            rows = rows.loc[rows['split'].eq(args.split)]
        if args.limit:
            rows = rows.head(args.limit)
        if rows.empty:
            raise ValueError('No input rows selected.')
        result = predict(args.model_dir, rows, args.threshold)
        report = evaluate(rows, result) if args.evaluate else None
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                result.to_csv(stream, index=False)
            print(f'Saved {len(result):,} predictions to {args.output}', file=sys.stderr)
        else:
            result.to_csv(sys.stdout, index=False)
        if report is not None:
            print('Batch metrics (confusion matrix: rows=actual, columns=predicted; order=[0, 1]):', file=sys.stderr)
            print(json.dumps(report, indent=2, allow_nan=False), file=sys.stderr)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
