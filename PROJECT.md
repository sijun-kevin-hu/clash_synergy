# Clash Synergy

## Purpose and research question
Georgia Tech CS4644 project by Kevin Hu, Ethan Ruan, and Joseph Velasquez.
Predict P(Player 1 wins) before a Trophy Road 1v1 battle. Test whether explicit
within-deck synergies and cross-deck counters improve prediction over XGBoost
and a neural network that encodes each deck independently.

## Planned comparison
1. XGBoost on fixed, order-independent features.
2. Independent-deck NN: shared card embeddings/encoder, separate pooling for each
   deck, concatenate representations, then an MLP. No explicit attention.
3. Proposed NN: within-deck self-attention and cross-deck attention, pooled deck
   representations, then a prediction head. No card-slot positional encoding.

Use binary cross-entropy for neural training. Report accuracy, ROC-AUC,
log loss/binary cross-entropy, and Brier score. Run deck-only and
with-player-statistics variants under the same splits and feature availability.
Eventually ablate self-attention and cross-attention separately. Attention alone
is not proof of causal synergy; compare learned patterns to empirical evidence.

## Verified pipeline
Repository: /Users/kevinhu/Programming/clash_synergy
Preprocessing: preprocessing/preprocessing.py (inspected 2026-11-05 ET).

```bash
python3 -m preprocessing.preprocessing \
  --input data/raw/combined_battles.jsonl \
  --output-dir data/processed/combined
```

Outputs:
- matches.jsonl: structured decks/levels, labels, metadata, histories and split.
- baseline.csv: fixed card-ID columns with level/maxLevel; absent card = 0;
  optional evolution/tower features; pre-match trophies and observed histories.
- card_catalog.json: API ID-to-name lookup, not trained embeddings.
- quality_report.json: filtering, date boundaries, split and label counts.

User-reported completed run (not a model result): 517,274 raw records; 24 ties
and 243 duplicates excluded; 517,007 usable matches; train 413,604;
validation 51,702; test 51,701; labels 0=258,604, 1=258,403;
123 unique cards and 380,516 unique players.
Battle range: 2026-09-01T00:00:09Z through 2026-11-05T23:53:18Z.
Training ends exclusively at 2026-11-05T02:58:39Z; validation ends exclusively
at 2026-11-05T14:16:08Z. Equal-time battles stay together.
Most data lies near the end of this date range; document the narrow validation
and test windows rather than implying an even temporal spread.

Player sides are assigned deterministically using seed 42 and match identity.
Histories contain only strictly earlier OBSERVED matches; no same-time leakage.
They update during validation/test, representing rolling online evaluation.
Zero-history win rate stays missing, not zero. Freeze input versions before
comparisons: rerunning preprocessing on more data shifts split boundaries.

## XGBoost implementation
models/train_xgboost.py reads the existing CSV directly, preserves splits,
allowlists features, and restricts card vocabulary to IDs present in training.
Default deck-only features are normalized card levels. Player-stats mode adds
starting trophies, trophy_diff, prior observed games and prior observed win rate.
Optional evolution/tower features require --include-optional; control their
availability consistently across models. Unknown cards are ignored and recorded
as excluded columns; this is a baseline limitation, not learned generalization.
Missing numeric values remain NaN for XGBoost. Scaling is unnecessary here.

```bash
python3 -m pip install 'xgboost>=2.1,<4' 'scikit-learn>=1.4,<2' 'pandas>=2.0' numpy
python3 models/train_xgboost.py --features deck-only
python3 models/train_xgboost.py --features with-player-stats
```

Defaults: hist tree method; depth 4; learning rate .05; at most 1,000 boosting
rounds; early stopping after 30 rounds without validation log-loss improvement;
row/column subsampling .8; seed 42; four CPU threads.
Outputs under artifacts/xgboost/<feature-mode>/ (optional features add -optional):
model.json, features.json, metrics.json, learning_curve.json.
Repeated runs with the same output directory replace its outputs; use distinct
--output-dir paths for experiments. Freeze choices before adding --evaluate-test.
For prediction, load features.json, reindex numerical inputs to that exact order,
preserve unknown values as NaN, and use predict_proba(X)[:, 1]. Construct absent
card values as zero using the same preprocessing semantics.

## Milestone
Instructions require evaluated baseline results and implementation evidence for
the proposed method; proposed-method results are optional. Include dataset and
pipeline details, technical approach, metrics/protocol, results table and next
steps. Use ICLR template, 4–5 main-text pages under the rubric, references excluded,
at least five related-work citations per rubric, and a teaser/draft figure.
Use the rubric corresponding to whether any member is in CS7643.

## Status and next steps
- Data collection/preprocessing exist; completed counts above came from terminal.
- XGBoost script prepared; no full-dataset trained scores established yet.
- Independent NN and attention-model training remain planned, not verified.
- Install staged files into the repository, install dependencies, train both
  baseline variants, examine validation metrics and overfitting, then freeze
  settings and evaluate test. Do not fabricate results for the milestone.
- Proposal also names a historical Season 18 dataset. Current progress concerns
  collected 2026 data; do not imply that Season 18 experiments are complete or
  combine old/new game environments without a separate documented protocol.

## Maintenance
Update this file when an experiment completes, a protocol changes, or new results
are verified. Record settings, dataset hash, output paths, metrics and remaining
limitations. Keep AGENT.md for durable working conventions.

## Delivery verification
2026-11-05 output fix: XGBoost's default `missing=NaN` parameter is recorded as
JSON `null` with an explicit encoding note, allowing strict metrics serialization
without changing training's missing-value behavior. Two regression checks passed
using local XGBoost 3.4.1, including a two-round synthetic CLI training run,
strict JSON artifact parsing, and model reload/prediction. Existing experiment
artifacts were not overwritten and full-dataset training was not rerun.

Historical preparation notes (superseded for local installation/smoke testing):
Prepared source passes Python compilation and synthetic loader/metric checks:
feature allowlist and leakage exclusions, deck-only versus player-statistics
inputs, training-only vocabulary, side symmetry, missing-value preservation,
split preservation, metric calculations, and invalid-split rejection.
XGBoost is not installed in the execution environment; network-restricted package
installation failed. Estimator training/save-load and full-data performance have
NOT been tested. Files are staged outside the repo because this session's write
permissions do not include the project path. Installation into the repo remains
necessary; do not claim these files are already active project instructions.
