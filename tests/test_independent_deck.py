import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn

    from models.independent_deck import IndependentDeckNet
    from models.train_independent_deck import (
        build_vocab,
        fit_player_stat_transform,
        load_matches,
        map_card_ids,
    )
except ImportError as exc:
    if getattr(exc, "name", None) not in (None, "torch") and "torch" not in str(exc):
        raise
    torch = None


def setUpModule():
    if torch is None:
        raise unittest.SkipTest("torch is not installed")

ROOT = Path(__file__).resolve().parents[1]


def side(card_ids, trophy=6000, games=4, win_rate=0.5):
    return {
        "player_tag": "#PLAYER",
        "starting_trophies": trophy,
        "prior_observed_games": games,
        "prior_observed_win_rate": win_rate,
        "cards": [
            {"id": int(card_id), "name": str(card_id), "level": 11, "maxLevel": 14}
            for card_id in card_ids
        ],
    }


def match(split, label, p1_ids, p2_ids, p1_win_rate=0.5, p2_win_rate=0.25):
    return {
        "match_id": f"{split}-{label}",
        "battle_time": "2026-01-01T00:00:00+00:00",
        "split": split,
        "label": label,
        "p1": side(p1_ids, 6400, 5, p1_win_rate),
        "p2": side(p2_ids, 6100, 5, p2_win_rate),
    }


def write_matches(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def tiny_dataset():
    rows = []
    for split in ("train", "validation", "test"):
        for index in range(12):
            label = index % 2
            p1 = list(range(1, 9)) if label else list(range(9, 17))
            p2 = list(range(9, 17)) if label else list(range(1, 9))
            rows.append(match(split, label, p1, p2, None if index == 0 else 0.4, 0.6))
    rows.append(match("validation", 1, [999, *range(2, 9)], list(range(9, 17))))
    return rows


class IndependentDeckTests(unittest.TestCase):
    def test_forward_shapes(self):
        deck_only = IndependentDeckNet(vocab_size=6, embed_dim=4, encoder_hidden=4, encoder_out=4, mlp_hidden=4, dropout=0)
        card_ids = torch.randint(1, 6, (4, 2, 8))
        levels = torch.rand(4, 2, 8)
        logits = deck_only(card_ids, levels)
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertTrue(torch.isfinite(logits).all())

        with_stats = IndependentDeckNet(
            vocab_size=6, embed_dim=4, encoder_hidden=4, encoder_out=4, mlp_hidden=4,
            player_stat_dim=9, dropout=0,
        )
        stats = torch.randn(4, 9)
        self.assertEqual(tuple(with_stats(card_ids, levels, stats).shape), (4,))
        with self.assertRaises(ValueError):
            deck_only(card_ids, levels, stats)
        with self.assertRaises(ValueError):
            with_stats(card_ids, levels)

    def test_predictions_ignore_card_order(self):
        torch.manual_seed(0)
        model = IndependentDeckNet(vocab_size=10, embed_dim=8, encoder_hidden=8, encoder_out=8, mlp_hidden=8, dropout=0.2)
        model.eval()
        card_ids = torch.randint(1, 10, (5, 2, 8))
        levels = torch.rand(5, 2, 8)
        order = torch.randperm(8)
        reordered_ids = card_ids.clone()
        reordered_levels = levels.clone()
        reordered_ids[:, 0] = card_ids[:, 0, order]
        reordered_levels[:, 0] = levels[:, 0, order]
        reordered_ids[:, 1] = card_ids[:, 1, order.flip(0)]
        reordered_levels[:, 1] = levels[:, 1, order.flip(0)]
        with torch.no_grad():
            original = model(card_ids, levels)
            shuffled = model(reordered_ids, reordered_levels)
        self.assertTrue(torch.allclose(original, shuffled, atol=1e-5))

    def test_player_order_changes_the_prediction(self):
        model = IndependentDeckNet(vocab_size=3, embed_dim=2, encoder_hidden=2, encoder_out=2, mlp_hidden=2, dropout=0)
        with torch.no_grad():
            model.embedding.weight.zero_()
            model.embedding.weight[1] = torch.tensor([1.0, 0.0])
            model.embedding.weight[2] = torch.tensor([0.0, 1.0])
            first, second = [layer for layer in model.card_encoder if isinstance(layer, nn.Linear)]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 0] = 1
            first.weight[1, 1] = 1
            second.weight.copy_(torch.eye(2))
            second.bias.zero_()
            hidden, output = [layer for layer in model.mlp if isinstance(layer, nn.Linear)]
            hidden.weight.zero_()
            hidden.bias.zero_()
            hidden.weight[0, 0] = 1
            hidden.weight[1, 2] = 1
            output.weight.zero_()
            output.bias.zero_()
            output.weight[0, 0] = 1
            output.weight[0, 1] = -1
        card_ids = torch.tensor([[[1] * 8, [2] * 8]])
        levels = torch.ones(1, 2, 8)
        model.eval()
        with torch.no_grad():
            self.assertAlmostEqual(model(card_ids, levels).item(), 1.0, places=5)
            self.assertAlmostEqual(model(card_ids.flip(1), levels).item(), -1.0, places=5)

    def test_vocabulary_and_imputation_use_training_rows_only(self):
        card_ids = np.array([
            [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 9]],
            [[99, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
        ], dtype=np.int64)
        splits = np.array([0, 1], dtype=np.int8)
        vocab = build_vocab(card_ids, splits)
        mapped = map_card_ids(card_ids, vocab)
        self.assertNotIn(99, vocab)
        self.assertEqual(mapped[1, 0, 0], 0)
        self.assertGreater(mapped[0, 1, 7], 0)

        stats = np.array([
            [10, 10, 0, 2, 2, 0.0, 0.0, 0, 0],
            [10, 10, 0, 2, 2, 0.0, 0.0, 0, 0],
            [10, 10, 0, 2, 2, 1.0, 1.0, 0, 0],
        ], dtype=np.float32)
        transform = fit_player_stat_transform(stats, np.array([True, True, False]))
        self.assertEqual(transform["impute"]["p1_prior_observed_win_rate"], 0.0)
        self.assertEqual(transform["impute"]["p2_prior_observed_win_rate"], 0.0)

    def test_loader_keeps_test_unread_until_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.jsonl"
            write_matches(path, tiny_dataset())
            held_out = load_matches(path, "deck-only", include_test=False)
            self.assertEqual(set(held_out["split"].tolist()), {0, 1})
            self.assertGreater(held_out["unread_test_rows"], 0)
            self.assertIsNone(held_out["player_stats"])
            scored = load_matches(path, "with-player-stats", include_test=True)
            self.assertIn(2, set(scored["split"].tolist()))
            self.assertEqual(scored["player_stats"].shape[1], 9)
            broken = match("train", 1, list(range(1, 8)), list(range(9, 17)))
            write_matches(path, [broken])
            with self.assertRaises(ValueError):
                load_matches(path, "deck-only", include_test=False)

    def test_training_writes_reloadable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "matches.jsonl"
            write_matches(source, tiny_dataset())
            result = subprocess.run([
                sys.executable, str(ROOT / "models" / "train_independent_deck.py"),
                "--input", str(source),
                "--output-dir", str(root / "output"),
                "--features", "with-player-stats",
                "--epochs", "2",
                "--patience", "2",
                "--batch-size", "8",
                "--embed-dim", "8",
                "--encoder-hidden", "8",
                "--encoder-out", "8",
                "--mlp-hidden", "8",
                "--threads", "1",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "output" / "with-player-stats"
            report = json.loads((output / "metrics.json").read_text())
            curve = json.loads((output / "learning_curve.json").read_text())
            config = json.loads((output / "config.json").read_text())
            self.assertIn("validation", report)
            self.assertNotIn("test", report)
            self.assertEqual(set(report["validation"]), {"accuracy", "roc_auc", "log_loss", "brier_score"})
            self.assertTrue(curve)
            self.assertEqual(config["architecture"]["player_stat_dim"], 9)
            state = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
            restored = IndependentDeckNet(**config["architecture"])
            restored.load_state_dict(state)
            restored.eval()
            logits = restored(torch.randint(1, 4, (2, 2, 8)), torch.rand(2, 2, 8), torch.randn(2, 9))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
