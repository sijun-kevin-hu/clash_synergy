"""Train the Clash Synergy baseline from preprocessing's baseline.csv.

Run from the repository root:
    python models/train_xgboost.py --features deck-only
    python models/train_xgboost.py --features with-player-stats
Add --evaluate-test only after model settings have been selected on validation.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

CARD = re.compile(r"p[12]_card_(\d+)_(level|evolution)$")
SUPPORT = re.compile(r"p[12]_support_(\d+)_level$")
PLAYER = {
    f"{side}_{field}"
    for side in ("p1", "p2")
    for field in ("starting_trophies", "prior_observed_games", "prior_observed_win_rate")
} | {"trophy_diff"}


def load_data(path, mode, include_optional):
    """Allowlist inputs so metadata and future outcomes cannot become features."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if not {"label", "split"}.issubset(header):
        raise ValueError("Input must contain label and split from preprocessing.")
    features = [c for c in header if (
        (CARD.fullmatch(c) and (c.endswith('_level') or include_optional))
        or (include_optional and SUPPORT.fullmatch(c))
        or (mode == 'with-player-stats' and c in PLAYER)
    )]
    if not features:
        raise ValueError("No recognized feature columns found.")
    if mode == 'with-player-stats' and not PLAYER.issubset(header):
        raise ValueError(f"Missing player features: {sorted(PLAYER - set(header))}")
    # Read just needed numeric columns as float32 to reduce memory on 500k+ rows.
    frame = pd.read_csv(path, usecols=features + ['label', 'split'],
                        dtype={c: 'float32' for c in features})
    if not frame['label'].isin([0, 1]).all():
        raise ValueError("Every label must be 0 or 1.")
    if not frame['split'].isin(['train', 'validation', 'test']).all():
        raise ValueError("Unexpected split name.")
    if np.isinf(frame[features].to_numpy()).any():
        raise ValueError("Features contain infinity; missing values may be NaN.")
    for name in ('train', 'validation', 'test'):
        labels = frame.loc[frame['split'].eq(name), 'label']
        if labels.empty or labels.nunique() != 2:
            raise ValueError(f"{name} must be nonempty and contain both labels.")

    # The CSV header includes all observed cards. Restrict model vocabulary to
    # card IDs actually present on EITHER side in training, preserving symmetry.
    training = frame['split'].eq('train')
    known_cards, known_support = set(), set()
    for column in features:
        match = CARD.fullmatch(column)
        support = SUPPORT.fullmatch(column)
        if match and match[2] == 'level' and frame.loc[training, column].gt(0).any():
            known_cards.add(match[1])
        if support and frame.loc[training, column].gt(0).any():
            known_support.add(support[1])
    kept = [c for c in features if (
        (CARD.fullmatch(c) and CARD.fullmatch(c)[1] in known_cards)
        or (SUPPORT.fullmatch(c) and SUPPORT.fullmatch(c)[1] in known_support)
        or c in PLAYER
    )]
    if not kept:
        raise ValueError("No features remain after training-only vocabulary selection.")
    dropped = sorted(set(features) - set(kept))
    frame = frame[['label', 'split'] + kept]
    return frame, kept, dropped


def metrics(y, probabilities):
    return {
        'accuracy': float(accuracy_score(y, probabilities >= 0.5)),
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'log_loss': float(log_loss(y, probabilities, labels=[0, 1])),
        'brier_score': float(brier_score_loss(y, probabilities)),
    }


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def report_parameters(model):
    """Represent XGBoost's NaN missing-value sentinel as JSON null."""
    parameters = model.get_params().copy()
    if np.isnan(parameters['missing']):
        parameters['missing'] = None
    return parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/combined/baseline.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/xgboost'))
    parser.add_argument('--features', choices=['deck-only', 'with-player-stats'], default='deck-only')
    parser.add_argument('--include-optional', action='store_true', help='Include recorded evolution/tower features; default uses card levels only.')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate test split after settings are fixed.')
    parser.add_argument('--n-estimators', type=int, default=1000)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--early-stopping-rounds', type=int, default=30)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.n_estimators, args.max_depth, args.early_stopping_rounds, args.threads) < 1 or not 0 < args.learning_rate <= 1:
        parser.error('Counts must be positive and learning rate must be in (0, 1].')

    print(f'Loading {args.input} ...', flush=True)
    frame, features, dropped = load_data(args.input, args.features, args.include_optional)
    def split(name):
        rows = frame['split'].eq(name)
        return frame.loc[rows, features], frame.loc[rows, 'label'].astype('int8')
    X_train, y_train = split('train')
    X_val, y_val = split('validation')
    print(f'Train={len(y_train):,}; validation={len(y_val):,}; features={len(features)}', flush=True)

    # XGBoost handles NaNs directly. Do not fill unknown history/evolution with 0.
    model = xgb.XGBClassifier(
        objective='binary:logistic', tree_method='hist',
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        learning_rate=args.learning_rate, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed, n_jobs=args.threads,
    )
    print('Training (validation log loss printed every 25 rounds) ...', flush=True)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)
    report = {
        'feature_mode': args.features, 'include_optional': args.include_optional,
        'feature_count': len(features), 'dropped_unseen_card_columns': dropped,
        'unknown_card_policy': 'Ignore card/support IDs absent from training; never add columns at prediction time.',
        'split_counts': {str(k): int(v) for k, v in frame['split'].value_counts().items()},
        'best_iteration_zero_based': int(model.best_iteration),
        'parameters': report_parameters(model),
        'parameter_encoding': 'parameters.missing=null represents the XGBoost NaN missing-value sentinel.',
        'versions': {'xgboost': xgb.__version__, 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'validation': metrics(y_val, model.predict_proba(X_val)[:, 1]),
        'training': metrics(y_train, model.predict_proba(X_train)[:, 1]),
        'history_policy': 'Strictly earlier observed matches; rolling history updates during validation/test.',
        'input': str(args.input.resolve()), 'input_sha256': digest(args.input),
        'feature_semantics': 'Card level / maxLevel; absent card = 0. Missing history/evolution = NaN. No feature scaling.',
    }
    # An inexpensive reference using only the training label distribution.
    prior = float(y_train.mean())
    report['constant_probability_validation'] = metrics(y_val, np.full(len(y_val), prior))
    if args.evaluate_test:
        X_test, y_test = split('test')
        report['test'] = metrics(y_test, model.predict_proba(X_test)[:, 1])
    quality = args.input.parent / 'quality_report.json'
    if quality.exists():
        report['preprocessing_report'] = json.loads(quality.read_text())

    suffix = '-optional' if args.include_optional else ''
    output = args.output_dir / (args.features + suffix)
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / 'model.json')
    (output / 'features.json').write_text(json.dumps(features, indent=2) + '\n')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'learning_curve.json').write_text(json.dumps(model.evals_result(), indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('training', 'validation', 'test') if k in report}, indent=2))
    print(f'COMPLETE: saved model, feature order, metrics and learning curve to {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
