"""Run saved XGBoost inference on CSV rows encoded by preprocessing."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss, confusion_matrix,
    precision_recall_fscore_support,
)
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def save_predictions(result, requested):
    """Keep the requested name first, then increase the largest saved suffix."""
    requested.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(requested.stem) + r'-(\d+)' + re.escape(requested.suffix))
    numbers = [int(match[1]) for path in requested.parent.iterdir()
               if (match := pattern.fullmatch(path.name))]
    number = max(numbers, default=0)
    candidate = requested if not numbers else requested.with_name(
        f'{requested.stem}-{number + 1}{requested.suffix}')
    while True:
        try:
            stream = candidate.open('x')
        except FileExistsError:
            number += 1
            candidate = requested.with_name(f'{requested.stem}-{number}{requested.suffix}')
            continue
        with stream:
            result.to_csv(stream, index=False)
        return candidate


def evaluate(rows, predictions):
    if 'label' not in rows or not rows['label'].isin([0, 1]).all():
        raise ValueError('--evaluate requires a label column containing only 0 or 1.')
    y = rows['label'].to_numpy()
    p = predictions['p1_win_probability'].to_numpy()
    predicted = predictions['predicted_p1_win'].to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1], zero_division=0)
    return {
        'classification': [
            {'precision': float(precision[i]), 'recall': float(recall[i]),
             'f1': float(f1[i]), 'support': int(support[i])} for i in range(2)
        ],
        'macro_f1': float(f1.mean()),
        'matches': len(y),
        'accuracy': float(accuracy_score(y, predicted)),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y, p)),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def show_report(console, report, model_dir, split, threshold):
    scores = Table(expand=True, header_style='bold cyan')
    scores.add_column('Metric')
    scores.add_column('Score', justify='right', style='bold')
    scores.add_column('Interpretation', overflow='fold')
    for label, value, note in (
        ('Accuracy', f"{report['accuracy']:.2%}", 'Correct winner predictions ↑'),
        ('Macro F1', f"{report['macro_f1']:.4f}", 'Mean F1 across both classes ↑'),
        ('ROC-AUC', 'N/A' if report['roc_auc'] is None else f"{report['roc_auc']:.4f}", 'Ranking quality ↑; needs both classes'),
        ('Log loss (BCE)', f"{report['log_loss']:.4f}", 'Probability error ↓'),
        ('Brier score', f"{report['brier_score']:.4f}", 'Mean squared probability error ↓'),
    ):
        scores.add_row(label, value, note)
    classes = Table(title='Classification report', expand=True, header_style='bold magenta')
    for name in ('Actual winner', 'Precision', 'Recall', 'F1', 'Support'):
        classes.add_column(name, justify='left' if name == 'Actual winner' else 'right')
    for label, values in zip(('Player 2 (0)', 'Player 1 (1)'), report['classification']):
        classes.add_row(label, *(f"{values[k]:.4f}" for k in ('precision', 'recall', 'f1')), str(values['support']))
    matrix = Table(title='Confusion matrix', expand=True, header_style='bold cyan')
    for name in ('Actual / Predicted', 'Player 2 (0)', 'Player 1 (1)'):
        matrix.add_column(name)
    for label, counts in zip(('Player 2 (0)', 'Player 1 (1)'), report['confusion_matrix']):
        matrix.add_row(label, *(f'{count:,}' for count in counts))
    context = Text(f"Model: {model_dir}\nSplit: {split or 'all selected rows'} | Matches: {report['matches']:,} | Threshold: {threshold:g}")
    note = Text('Undefined precision/recall/F1 use 0. R² is a regression metric; this is binary classification.', style='dim')
    console.print(Panel(Group(context, scores, classes, matrix, note), title='Batch evaluation', border_style='cyan'))


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
    parser.add_argument('--output', type=Path, help='Save CSV, adding increasing numeric suffixes on repeat runs; otherwise print CSV to stdout.')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], help='Select rows by their existing split.')
    parser.add_argument('--limit', type=int, help='Predict only the first N rows after split selection.')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--evaluate', action='store_true', help='Print labeled batch metrics to stderr; ROC-AUC is null for a single-class batch.')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    console = Console(stderr=True, highlight=False)
    try:
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
            output = save_predictions(result, args.output)
            console.print(Text(f'Saved {len(result):,} predictions to {output}', style='green'))
        else:
            result.to_csv(sys.stdout, index=False)
        if report is not None:
            show_report(console, report, args.model_dir, args.split, args.threshold)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
