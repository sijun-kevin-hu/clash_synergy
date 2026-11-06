# ClashSynergy

Data pipeline for learning within-deck synergies and cross-deck counters for pre-match Trophy Road outcome prediction. Adapted from Kevin Hu's Clash Predict personal project; original scripts are preserved in `legacy/`. XGBoost training is provided in `models/train_xgboost.py`; independent-deck and attention neural networks are planned.

## Setup and collection

Run from this folder with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env.local
```

Edit `.env.local` with your API token and the API endpoint configured for that token. Credentials were not copied from the old project. The original proxy endpoint is retained as the configurable default; access has not been live-tested.

```bash
python -m data_collection.clashapi --seed-player '#YOUR_PLAYER_TAG'
# Alternatively, seed from one or more clans:
python -m data_collection.clashapi --seed-clan '#YOUR_CLAN_TAG'
```

Each invocation targets **250,000 additional unique qualifying battles**, regardless of existing file size. It can stop earlier if the queue is exhausted or the API fails. Duplicate battles do not count. Subsequent invocations resume pending players:

```bash
python -m data_collection.clashapi
```

An interactive terminal automatically shows a live neon dashboard: session progress,
battles/minute, estimated time remaining, player frontier, archive size, retries,
filtered/duplicate counts, and recent activity. The display refreshes while API
requests or backoff timers are running. Ctrl-C saves progress and restores your
terminal. Restart the collector to enable the dashboard on an existing run.
Use `--ui plain` for traditional logs; redirected output uses plain logs automatically.
The dashboard uses Rich (installed with the requirements) for cyan/magenta panels,
animated progress, API health counters, and recent activity. Narrow terminals stack
the metric panels; short terminals hide activity to prioritize collection status.

To refresh previously visited players for newer logs, use `--revisit`; supply extra seeds if needed. Use `--target-battles 1000` for a small trial. Do not run multiple collectors against the same raw directory simultaneously. Interrupt with Ctrl-C to save progress. State is written atomically; deduplication is rebuilt from raw records at startup. A truncated JSON line must be repaired before resuming.

Only exact `PvP` / `Ladder` 1v1 records are accepted. The old 3,000–9,000 trophy restriction is removed because the proposal specifies Trophy Road generally; both players' starting trophies must fall within the configurable range (default 0–100,000). The upper bound is a configurable sanity limit, not an asserted current game maximum. Validate current API mode identifiers with an initial small collection before scaling.

## Preprocessing

```bash
python -m preprocessing.preprocessing
```

To process the previous project's data without copying the raw file:

```bash
python -m preprocessing.preprocessing \
  --input /Users/kevinhu/Programming/clash_predict/data/raw/battles.jsonl
```

Generated under `data/processed/`:

- `matches.jsonl`: one match per line; raw card IDs/levels/max levels, optional evolution and elixir fields, tower troops, pre-match trophies, prior observed histories, timestamp, label and split. Cards sorted by ID for reproducibility; models must treat slots as a set without positional embeddings.
- `baseline.csv`: wide normalized card-level features and optional recorded evolution/tower features, plus trophies/history and metadata.
- `card_catalog.json`: observed API card IDs to names; not a learned embedding vocabulary.
- `quality_report.json`: filtering counts, split/label counts and date range.

Splits approximate 80/10/10 by chronological match count. Identical timestamps stay together, so proportions can differ and tiny datasets can have empty splits. Player assignment is deterministic pseudorandom using match ID and seed; it produces approximate, not guaranteed exact, label balance. Ties and invalid/incomplete decks are excluded. Reprocessing a larger raw file recomputes split boundaries; freeze dataset versions for comparable experiments.

Historical features are counts and win rates from **strictly earlier observed matches only**, not lifetime statistics. Equal-time matches cannot use one another's outcomes. Histories update throughout validation/test to simulate rolling pre-match evaluation; this assumption must be reported. No current profile statistics are fetched or substituted for past statistics. Zero-history win rates are null, not fabricated values.

## XGBoost training

Run commands from the repository root with your virtual environment active:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The training dependencies are `numpy`, `pandas`, `scikit-learn`, and `xgboost`.
Prepare the combined dataset if it has not already been preprocessed:

```bash
python3 -m preprocessing.preprocessing \
  --input data/raw/combined_battles.jsonl \
  --output-dir data/processed/combined
```

Train the two baseline variants independently:

```bash
python3 models/train_xgboost.py --features deck-only
python3 models/train_xgboost.py --features with-player-stats
```

Both read `data/processed/combined/baseline.csv` by default. Deck-only uses
per-card normalized levels (`level / maxLevel`), with zero indicating an absent
card. With-player-stats also uses starting trophies, trophy difference, prior
observed games, and prior observed win rate. Raw metadata, identifiers and outcomes
are excluded from model features. Unknown histories stay missing, not zero.

The existing chronological train/validation/test assignments are preserved.
Only training rows fit trees. Validation log loss controls early stopping;
training and validation accuracy, ROC-AUC, log loss and Brier score are reported.
Historical features use strictly earlier observed matches, including rolling
history updates during validation/test. No feature scaling is applied.

Default settings: up to 1,000 boosting rounds, tree depth 4, learning rate 0.05,
row/column subsampling 0.8, four CPU threads, seed 42, and early stopping after
30 rounds without validation improvement. The terminal prints validation loss
every 25 rounds, followed by metrics and `COMPLETE` once all outputs are saved.

Outputs are separated by variant:

| Directory contents | Purpose |
| --- | --- |
| `model.json` | Trained XGBoost model |
| `features.json` | Exact feature-column order required for inference |
| `metrics.json` | Metrics, settings, versions, input hash and dataset summary |
| `learning_curve.json` | Validation log loss by boosting round |

Default directories are `artifacts/xgboost/deck-only/` and
`artifacts/xgboost/with-player-stats/`. Reusing an output directory replaces
its saved files. Use a separate output root for each experiment:

```bash
python3 models/train_xgboost.py \
  --features deck-only \
  --max-depth 3 \
  --learning-rate 0.05 \
  --output-dir artifacts/xgboost/depth3
```

This example writes to `artifacts/xgboost/depth3/deck-only/`.
Use `--input PATH` for a different preprocessed CSV. Add `--include-optional`
to include recorded evolution and tower-troop features; these runs add
`-optional` to the variant directory. Compare models using consistent optional
inputs. Run `python3 models/train_xgboost.py --help` for all settings.

Test metrics are deliberately omitted by default. After selecting settings on
validation, rerun the selected configuration with `--evaluate-test` to report
final test performance; this command trains again before evaluation. For example:

```bash
python3 models/train_xgboost.py \
  --features deck-only \
  --evaluate-test \
  --output-dir artifacts/xgboost/final
```

Preserve the same frozen dataset and selected hyperparameters. Do not tune
against repeated test evaluations.

## XGBoost inference

Inference loads a saved model and returns `P(Player 1 wins)` without training.
The default command evaluates the entire validation split with the deck-only
model, displays Rich metric tables, and saves automatically numbered predictions:

```bash
python3 models/predict_xgboost.py
```

No flags are required. Optional flags override these defaults. For unlabeled
custom inputs use `--split all --no-evaluate`; use `--output` to choose a save path.

Run from the repository root with your virtual environment active:

```bash
# Preview five predictions using the deck-only model.
python3 models/predict_xgboost.py --limit 5

# Use the player-statistics model and save validation predictions.
python3 models/predict_xgboost.py \
  --model-dir artifacts/xgboost/with-player-stats \
  --split validation \
  --output predictions/player-stats-validation.csv

# Predict new matchups already encoded with the preprocessing CSV schema.
python3 models/predict_xgboost.py \
  --input new_matchups.csv \
  --output predictions/new-matchups.csv
```

Use `--model-dir` for any saved experiment directory containing `model.json`
and `features.json`. `--limit N` applies after `--split` filtering; without a
split selection the input does not need a split column. `--threshold` defaults
to 0.5. Output contains optional match_id, p1_win_probability and predicted_p1_win.
By default, CSV is saved to `predictions/validation-report.csv`. Repeat runs automatically add increasing suffixes (`validation.csv`,
`validation-1.csv`, `validation-2.csv`, …), continuing above the largest existing
suffix and preserving previous files. The saved path is printed in the terminal.
Required feature columns are validated and reordered; extra metadata is excluded,
and NaN values are preserved. These commands do not train. Evaluation is enabled by default to show a Rich evaluation panel with accuracy, macro F1, ROC-AUC,
log loss (BCE), Brier score, per-class precision/recall/F1/support and a labeled
confusion matrix on stderr using the selected rows
and their labels. ROC-AUC is null when only one class is selected. For example:

```bash
python3 models/predict_xgboost.py --split validation --limit 1000 --evaluate --output predictions/validation-1000.csv
```
Default rows may include training examples; use validation for development and
reserve test evaluation for your final selected model.

Equivalent Python example:

```bash
python3 - <<'PY'
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

# Change this path to select the trained variant or experiment.
model_dir = Path("artifacts/xgboost/deck-only")
features = json.loads((model_dir / "features.json").read_text())

model = XGBClassifier()
model.load_model(model_dir / "model.json")

# Replace this input with matches encoded using the same preprocessing schema.
rows = pd.read_csv("data/processed/combined/baseline.csv", nrows=5)
missing = set(features) - set(rows.columns)
if missing:
    raise ValueError(f"Input is missing required feature columns: {sorted(missing)}")

# Load only saved numerical features, in precisely their training order.
# Metadata, split and label are never passed to the model.
X = rows.reindex(columns=features).astype("float32")
probabilities = model.predict_proba(X)[:, 1]

predictions = pd.DataFrame({
    "p1_win_probability": probabilities,
    "predicted_p1_win": (probabilities >= 0.5).astype(int),
})
if "match_id" in rows.columns:
    predictions.insert(0, "match_id", rows["match_id"].to_numpy())
print(predictions.to_string(index=False))
PY
```

A probability of `0.72` means an estimated 72% chance that the player encoded
as Player 1 wins. A threshold of 0.5 converts probabilities into winner labels.

For new matchups, provide **pre-match features in the same encoding**: card
levels divided by maxLevel; absent known cards set to zero; and, when required,
pre-match trophies and strictly prior observed histories. Keep unknown history
and unrecorded evolution as NaN. A completely absent feature column is a schema
error in this example, so explicitly construct all required columns. Card IDs
never observed in training are ignored by the saved feature list; the model
cannot learn effects for those cards. Optional evolution/tower inputs must match
the trained variant. Raw decks or card names alone cannot be passed directly to
`predict_proba`; they first need the matching feature transformation.

## Model integration rules

Never feed `match_id`, `battle_time`, player tags, `split`, or `label` into a model. Fit imputers, scalers and any learned card vocabulary on the training split only, with an unknown-card policy for validation/test. For a deck-only experiment, exclude trophies and historical features; compare optional player-information variants consistently across all three models. Do not use labels/crowns or current player profiles as pre-match features. Raw crowns are used only to form labels and earlier observed win histories.

Use the same splits and feature availability for XGBoost, the independent-deck network and attention model. Decide whether tower troops/evolution are shared inputs or controlled exclusions; the new pipeline preserves available values but does not invent missing ones. An absent card has zero baseline level; a present card with unknown evolution has a blank evolution value. Missing elixir is not treated as zero. Manual role maps are optional legacy features and are not required for learned interactions.

The copied `utils/fetch_cards.py` and cached card/role metadata support the original preprocessing; the new pipeline runs offline and does not require them. The caches are historical, not verified current. The old `legacy/preprocessing_original.py` expects its original package setup; it is preserved for reference, not the new project's entry point.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover filtering, deduplication, labels after side assignment, chronological separation, strict prior histories, equal-time isolation, and the collector's per-invocation target/resume behavior with mocked API responses. The full existing raw dataset was also processed successfully. No live collection has been started.

See `reports/DATA_FIT_REPORT.md` for the data audit and remaining limitations.

## Frozen static reference data

RoyaleAPI card metadata, rarity/mode catalogs and combat stats are stored in
`data/static/royaleapi/`. See its README and manifest for the pinned version,
source URLs and checksums. The latest upstream commit available at download was
October 18, 2023; these files are **not verified current** for November 2026
battles. They are reference data and are not automatically added to model inputs.
