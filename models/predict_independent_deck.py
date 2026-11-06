"""Run saved independent-deck inference on matches.jsonl rows.

Does not train. Loads model.pt and config.json, then returns P(Player 1 wins).
Card ids and player-stat normalization come from that checkpoint, not from the
rows being scored.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score,
    precision_recall_fscore_support,
)
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.independent_deck import IndependentDeckNet
from models.train_independent_deck import deck_arrays, raw_player_stats, transform_player_stats


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
            stream = candidate.open('x', encoding='utf-8', newline='')
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


def load_checkpoint(model_dir, device):
    model_dir = Path(model_dir)
    config_path = model_dir / 'config.json'
    weights_path = model_dir / 'model.pt'
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f'{model_dir} must contain config.json and model.pt.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    architecture = config.get('architecture')
    if not isinstance(architecture, dict) or 'player_stat_dim' not in architecture:
        raise ValueError('config.json is missing the saved architecture.')
    model = IndependentDeckNet(**architecture)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


def select_rows(path, split, limit):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Could not find {path}.')
    selected = []
    seen_split = False
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                match = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {line_number}: {exc}') from exc
            if 'split' in match:
                seen_split = True
            if split is not None and match.get('split') != split:
                continue
            selected.append(match)
            if limit is not None and len(selected) >= limit:
                break
    if split is not None and not seen_split:
        raise ValueError('--split requires a split field in the input matches.')
    if not selected:
        raise ValueError('No input rows selected.')
    return selected


def encode_rows(rows, config):
    """Map decks with the checkpoint vocabulary. Do not refit normalization."""
    architecture = config['architecture']
    with_stats = int(architecture['player_stat_dim']) > 0
    transform = config.get('player_stat_transform')
    if with_stats and not transform:
        raise ValueError('with-player-stats checkpoint is missing player_stat_transform.')
    vocab = {int(card_id): int(index) for card_id, index in config['card_to_index'].items()}
    unknown = int(config.get('unknown_index', 0))
    card_rows, level_rows, stat_rows = [], [], []
    for index, row in enumerate(rows, 1):
        try:
            p1_ids, p1_levels = deck_arrays(row['p1'], 'p1')
            p2_ids, p2_levels = deck_arrays(row['p2'], 'p2')
            if with_stats:
                stat_rows.append(raw_player_stats(row))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid match at selected row {index}: {exc}') from exc
        card_rows.append(np.stack([p1_ids, p2_ids]))
        level_rows.append(np.stack([p1_levels, p2_levels]))
    raw_ids = np.stack(card_rows)
    flat = raw_ids.reshape(-1)
    mapped = np.fromiter((vocab.get(int(card_id), unknown) for card_id in flat), dtype=np.int64, count=flat.size)
    encoded = {
        'card_ids': mapped.reshape(raw_ids.shape),
        'levels': np.stack(level_rows),
        'player_stats': None,
    }
    if with_stats:
        encoded['player_stats'] = transform_player_stats(np.stack(stat_rows), transform)
    return encoded


def predict(model_dir, rows, threshold=0.5, batch_size=1024, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_checkpoint(model_dir, device)
    encoded = encode_rows(rows, config)
    card_ids = encoded['card_ids']
    levels = encoded['levels']
    stats = encoded['player_stats']
    if int(config['architecture']['vocab_size']) <= int(card_ids.max()):
        raise ValueError('A mapped card index is outside the saved embedding table.')
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = start + batch_size
            batch_ids = torch.from_numpy(card_ids[start:stop]).to(device)
            batch_levels = torch.from_numpy(levels[start:stop]).to(device)
            batch_stats = None if stats is None else torch.from_numpy(stats[start:stop]).to(device)
            logits = model(batch_ids, batch_levels, batch_stats)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    probabilities = np.concatenate(probabilities)
    result = pd.DataFrame({
        'p1_win_probability': probabilities,
        'predicted_p1_win': (probabilities >= threshold).astype('int8'),
    })
    if any('match_id' in row for row in rows):
        result.insert(0, 'match_id', [row.get('match_id') for row in rows])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/independent_deck_nn/deck-only'))
    parser.add_argument('--input', type=Path, default=Path('data/processed/combined/matches.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('predictions/independent-deck-validation.csv'), help='Save CSV, adding increasing numeric suffixes on repeat runs.')
    parser.add_argument('--split', choices=['train', 'validation', 'test', 'all'], default='validation', help='Select existing split (default: validation); all selects every row.')
    parser.add_argument('--limit', type=int, help='Predict only the first N rows after split selection.')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=True, help='Show batch metrics (enabled by default); --no-evaluate skips metrics for unlabeled inputs.')
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1.')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    split = None if args.split == 'all' else args.split
    console = Console(stderr=True, highlight=False)
    try:
        rows = select_rows(args.input, split, args.limit)
        result = predict(args.model_dir, rows, args.threshold, args.batch_size)
        report = None
        if args.evaluate:
            try:
                labels = [row['label'] for row in rows]
            except KeyError as exc:
                raise ValueError('--evaluate requires a label column containing only 0 or 1.') from exc
            report = evaluate(pd.DataFrame({'label': labels}), result)
        if args.output:
            output = save_predictions(result, args.output)
            console.print(Text(f'Saved {len(result):,} predictions to {output}', style='green'))
        else:
            result.to_csv(sys.stdout, index=False)
        if report is not None:
            show_report(console, report, args.model_dir, split, args.threshold)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
