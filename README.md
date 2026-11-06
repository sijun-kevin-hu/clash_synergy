# ClashSynergy

Data pipeline for learning within-deck synergies and cross-deck counters for pre-match Trophy Road outcome prediction. Adapted from Kevin Hu's Clash Predict personal project; original scripts are preserved in `legacy/`. Training code is not included yet.

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
