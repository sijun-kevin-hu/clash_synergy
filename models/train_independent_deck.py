"""Train the independent-deck neural baseline from preprocessing's matches.jsonl.

Run from the repository root:
    python models/train_independent_deck.py --features deck-only
    python models/train_independent_deck.py --features with-player-stats
Add --evaluate-test only after model settings have been selected on validation.

Reads data/processed/combined/matches.jsonl by default. Does not modify
preprocessing or the files it writes.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.independent_deck import IndependentDeckNet, count_parameters

CONTINUOUS_STATS = (
    "p1_starting_trophies",
    "p2_starting_trophies",
    "trophy_diff",
    "p1_prior_observed_games",
    "p2_prior_observed_games",
    "p1_prior_observed_win_rate",
    "p2_prior_observed_win_rate",
)
MISSING_FLAGS = ("p1_win_rate_missing", "p2_win_rate_missing")
PLAYER_STAT_NAMES = CONTINUOUS_STATS + MISSING_FLAGS
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
WIN_RATE_COLUMNS = {"p1_prior_observed_win_rate": 5, "p2_prior_observed_win_rate": 6}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def finite_number(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not a finite number.")
    return number


def deck_arrays(side, name):
    cards = side.get("cards")
    if not isinstance(cards, list) or len(cards) != 8:
        raise ValueError(f"{name} must contain exactly 8 cards.")
    card_ids = np.empty(8, dtype=np.int64)
    levels = np.empty(8, dtype=np.float32)
    for index, card in enumerate(cards):
        card_ids[index] = int(card["id"])
        max_level = finite_number(card["maxLevel"], f"{name} maxLevel")
        if max_level <= 0:
            raise ValueError(f"{name} maxLevel must be positive.")
        levels[index] = finite_number(card["level"], f"{name} level") / max_level
    return card_ids, levels


def raw_player_stats(match):
    """Pre-match trophies and strictly prior observed history. Null win rate stays missing."""
    p1, p2 = match["p1"], match["p2"]
    trophies = [
        finite_number(p1["starting_trophies"], "p1 starting_trophies"),
        finite_number(p2["starting_trophies"], "p2 starting_trophies"),
    ]
    games = [
        finite_number(p1["prior_observed_games"], "p1 prior_observed_games"),
        finite_number(p2["prior_observed_games"], "p2 prior_observed_games"),
    ]
    rates = []
    missing = []
    for side, player in (("p1", p1), ("p2", p2)):
        rate = player["prior_observed_win_rate"]
        if rate is None:
            rates.append(np.nan)
            missing.append(1.0)
        else:
            rates.append(finite_number(rate, f"{side} prior_observed_win_rate"))
            missing.append(0.0)
    return np.array([
        trophies[0], trophies[1], trophies[0] - trophies[1],
        games[0], games[1], rates[0], rates[1], missing[0], missing[1],
    ], dtype=np.float32)


def parse_match(match, with_stats):
    label = match.get("label")
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1.")
    split = match.get("split")
    if split not in SPLIT_CODE:
        raise ValueError(f"Unexpected split {split!r}.")
    p1_ids, p1_levels = deck_arrays(match["p1"], "p1")
    p2_ids, p2_levels = deck_arrays(match["p2"], "p2")
    card_ids = np.stack([p1_ids, p2_ids])
    levels = np.stack([p1_levels, p2_levels])
    stats = raw_player_stats(match) if with_stats else None
    return card_ids, levels, np.float32(label), SPLIT_CODE[split], stats


def build_vocab(card_ids, split_codes):
    """Map card ids observed in training to embedding indexes. Index 0 is unknown."""
    train_ids = card_ids[split_codes == 0].reshape(-1)
    ordered = sorted({int(card_id) for card_id in train_ids.tolist()})
    return {card_id: index for index, card_id in enumerate(ordered, start=1)}


def map_card_ids(card_ids, vocab):
    flat = card_ids.reshape(-1)
    mapped = np.fromiter((vocab.get(int(card_id), 0) for card_id in flat), dtype=np.int64, count=flat.size)
    return mapped.reshape(card_ids.shape)


def fit_player_stat_transform(stats, train_mask):
    """Impute and z-score from training rows only. Missingness flags stay 0/1."""
    train = stats[train_mask]
    impute = {}
    for name, column in WIN_RATE_COLUMNS.items():
        observed = train[:, column]
        observed = observed[np.isfinite(observed)]
        # No observed training history: use a neutral rate. The missingness flag
        # still tells the network that the value was not observed.
        impute[name] = 0.5 if observed.size == 0 else float(observed.mean())
    filled = apply_imputation(stats, impute)
    continuous = filled[train_mask, : len(CONTINUOUS_STATS)]
    mean = continuous.mean(axis=0)
    std = continuous.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "names": list(PLAYER_STAT_NAMES),
        "impute": impute,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "standardized_columns": len(CONTINUOUS_STATS),
    }


def apply_imputation(stats, impute):
    filled = np.array(stats, dtype=np.float32, copy=True)
    for name, column in WIN_RATE_COLUMNS.items():
        missing = ~np.isfinite(filled[:, column])
        filled[missing, column] = np.float32(impute[name])
    return filled


def transform_player_stats(stats, transform):
    transformed = apply_imputation(stats, transform["impute"])
    width = transform["standardized_columns"]
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)
    transformed[:, :width] = (transformed[:, :width] - mean) / std
    return transformed


def load_matches(path, feature_mode, include_test):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Run preprocessing first; this script does not create matches.jsonl."
        )
    with_stats = feature_mode == "with-player-stats"
    card_rows, level_rows, labels, splits = [], [], [], []
    stat_rows = []
    unread_test = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if line_number % 100_000 == 0:
                print(f"  read {line_number:,} lines", flush=True)
            try:
                match = json.loads(line)
                if match.get("split") == "test" and not include_test:
                    unread_test += 1
                    continue
                parsed = parse_match(match, with_stats)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid match on line {line_number}: {exc}") from exc
            card_rows.append(parsed[0])
            level_rows.append(parsed[1])
            labels.append(parsed[2])
            splits.append(parsed[3])
            if with_stats:
                stat_rows.append(parsed[4])
    if not card_rows:
        raise ValueError("No matches were loaded.")
    card_ids = np.stack(card_rows)
    levels = np.stack(level_rows)
    labels = np.asarray(labels, dtype=np.float32)
    splits = np.asarray(splits, dtype=np.int8)
    for name, code in (("train", 0), ("validation", 1)):
        require_both_labels(labels[splits == code], name)
    vocab = build_vocab(card_ids, splits)
    if not vocab:
        raise ValueError("Training split has no card ids.")
    mapped = map_card_ids(card_ids, vocab)
    stats = None
    transform = None
    if with_stats:
        stats = np.stack(stat_rows)
        transform = fit_player_stat_transform(stats, splits == 0)
        stats = transform_player_stats(stats, transform)
    if include_test:
        require_both_labels(labels[splits == 2], "test")
    return {
        "card_ids": mapped,
        "levels": levels,
        "labels": labels,
        "split": splits,
        "player_stats": stats,
        "vocab": vocab,
        "player_stat_transform": transform,
        "unread_test_rows": unread_test,
    }


def require_both_labels(labels, name):
    if labels.size == 0 or set(np.unique(labels).tolist()) != {0.0, 1.0}:
        raise ValueError(f"{name} must be nonempty and contain both labels.")


def classification_metrics(labels, probabilities):
    return {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(data, code, batch_size, shuffle, seed):
    selected = data["split"] == code
    tensors = [
        torch.from_numpy(data["card_ids"][selected]),
        torch.from_numpy(data["levels"][selected]),
        torch.from_numpy(data["labels"][selected]),
    ]
    if data["player_stats"] is not None:
        tensors.append(torch.from_numpy(data["player_stats"][selected]))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
        generator=generator, num_workers=0,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for batch in loader:
        batch = [tensor.to(device) for tensor in batch]
        stats = batch[3] if len(batch) == 4 else None
        card_ids, levels, labels = batch[:3]
        if training:
            optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(card_ids, levels, stats), labels)
        if training:
            loss.backward()
            optimizer.step()
        total += loss.item() * labels.shape[0]
        count += int(labels.shape[0])
    return total / count


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        batch = [tensor.to(device) for tensor in batch]
        stats = batch[3] if len(batch) == 4 else None
        logits = model(batch[0], batch[1], stats)
        labels.append(batch[2].cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train(args):
    set_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    print(f"Loading {args.input} ...", flush=True)
    data = load_matches(args.input, args.features, args.evaluate_test)
    device = select_device()
    player_stat_dim = 0 if data["player_stats"] is None else data["player_stats"].shape[1]
    model = IndependentDeckNet(
        vocab_size=len(data["vocab"]) + 1,
        embed_dim=args.embed_dim,
        encoder_hidden=args.encoder_hidden,
        encoder_out=args.encoder_out,
        mlp_hidden=args.mlp_hidden,
        player_stat_dim=player_stat_dim,
        dropout=args.dropout,
    ).to(device)
    parameter_count = count_parameters(model)
    train_count = int((data["split"] == 0).sum())
    val_count = int((data["split"] == 1).sum())
    print(
        f"Train={train_count:,}; validation={val_count:,}; cards={len(data['vocab'])}; "
        f"parameters={parameter_count:,}; device={device}",
        flush=True,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = make_loader(data, 0, args.batch_size, True, args.seed)
    val_loader = make_loader(data, 1, args.batch_size, False, args.seed)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    best_state = None
    curve = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        curve.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} validation_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = cpu_state(model)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best validation loss was epoch {best_epoch}.", flush=True)
                break
    model.load_state_dict(best_state)
    model.to(device)

    def score(code):
        observed, probabilities = predict_probabilities(
            model, make_loader(data, code, args.batch_size, False, args.seed), device,
        )
        return classification_metrics(observed, probabilities), observed

    train_metrics, train_labels = score(0)
    val_metrics, val_labels = score(1)
    prior = float(train_labels.mean())
    architecture = {
        "vocab_size": len(data["vocab"]) + 1,
        "embed_dim": args.embed_dim,
        "encoder_hidden": args.encoder_hidden,
        "encoder_out": args.encoder_out,
        "mlp_hidden": args.mlp_hidden,
        "dropout": args.dropout,
        "player_stat_dim": player_stat_dim,
    }
    config = {
        "feature_mode": args.features,
        "architecture": architecture,
        "parameter_count": parameter_count,
        "unknown_index": 0,
        "card_to_index": {str(card_id): index for card_id, index in sorted(data["vocab"].items())},
        "player_stat_transform": data["player_stat_transform"],
        "seed": args.seed,
        "level_semantics": "level / maxLevel. Mean pooling discards card order.",
        "unknown_card_policy": "Card IDs absent from the training split use embedding index 0.",
    }
    report = {
        "feature_mode": args.features,
        "parameter_count": parameter_count,
        "architecture": architecture,
        "best_epoch_1_based": best_epoch,
        "epochs_ran": len(curve),
        "best_validation_loss": best_loss,
        "split_counts": {
            "train": train_count,
            "validation": val_count,
        },
        "unread_test_rows": data["unread_test_rows"],
        "training": train_metrics,
        "validation": val_metrics,
        "constant_probability_validation": classification_metrics(val_labels, np.full(len(val_labels), prior)),
        "history_policy": "Strictly earlier observed matches from matches.jsonl; histories are not recomputed.",
        "player_stat_semantics": (
            "Excluded." if args.features == "deck-only" else
            "Starting trophies, trophy difference, prior games, and prior win rate. "
            "Null win rates are mean-imputed from training and marked with a missingness flag. "
            "Continuous stats are z-scored with training mean and standard deviation."
        ),
        "input": str(Path(args.input).resolve()),
        "input_sha256": digest(args.input),
        "versions": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "device": str(device),
    }
    if args.evaluate_test:
        test_count = int((data["split"] == 2).sum())
        report["split_counts"]["test"] = test_count
        report["test"], _ = score(2)
    output = Path(args.output_dir) / args.features
    output.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state(model), output / "model.pt")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "learning_curve.json").write_text(json.dumps(curve, indent=2) + "\n")
    shown = {key: report[key] for key in ("training", "validation", "test") if key in report}
    print(json.dumps(shown, indent=2))
    print(f"COMPLETE: saved model, config, metrics and learning curve to {output.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/combined/matches.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/independent_deck_nn"))
    parser.add_argument("--features", choices=["deck-only", "with-player-stats"], default="deck-only")
    parser.add_argument("--evaluate-test", action="store_true", help="Score the test split after settings are fixed.")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--encoder-hidden", type=int, default=64)
    parser.add_argument("--encoder-out", type=int, default=64)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many epochs without a better validation loss.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sizes = (args.embed_dim, args.encoder_hidden, args.encoder_out, args.mlp_hidden, args.epochs, args.patience, args.batch_size, args.threads)
    if min(sizes) < 1 or not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Sizes and patience must be positive, dropout must be in [0, 1), and learning rate must be positive.")
    try:
        train(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
