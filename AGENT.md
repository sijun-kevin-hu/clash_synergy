# Working conventions for Clash Synergy

Read PROJECT.md and README.md before making changes. This file was explicitly
requested as AGENT.md; tooling that auto-loads only AGENTS.md may require an
explicit reference to this file. Keep the requested filename.

## Collaboration
Kevin wants to learn implementation steps and why they work. Explain data flow,
model inputs, training, prediction and evaluation in plain language. Implement
requested changes and summarize meaningful verification. Keep PROJECT.md current
with verified progress; distinguish plans, smoke tests and full-dataset results.
Do not claim a model was trained based on preprocessing output alone.

## Data and evaluation invariants
- Only pre-match information is allowed. Never use current outcomes, crowns,
  post-match player information, raw player IDs, match IDs, split or timestamp
  as initial model features. Outcomes are targets and inputs to strictly prior
  observed history only.
- Preserve existing chronological splits; equal-time matches stay together.
- Fit transformations/vocabularies on training only; define unknown-card policy.
- Use validation for tuning/early stopping and test only for final evaluation.
- Histories use strictly earlier observed matches and roll forward during
  validation/test. State this policy in reports.
- Deck-only excludes trophies AND all player history. Keep optional evolution
  and tower inputs consistent across compared models.
- Preserve absent-card zero versus missing-history/evolution NaN semantics.
- Decks are unordered sets: no meaningful numeric ordering of IDs and no
  card-slot positional embeddings. Pool NN card representations per deck.
- Use the same datasets, available information and metrics for all comparisons.
- Never fetch current player statistics to backfill historical match features.

## Reproducibility and verification
Record input hash/version, split counts, feature order, random seed, settings,
dependency versions, and metrics. Keep each experiment in its own output path.
Report accuracy, ROC-AUC, binary cross-entropy/log loss, and Brier score.
Use train/validation learning behavior to assess overfitting. Do not infer causal
synergy solely from predictive gains or attention weights.
Run focused synthetic checks for feature leakage, split preservation and model
save/load when training logic changes; run existing tests for pipeline changes:

```bash
python3 -m unittest discover -s tests -v
```

Do not modify raw datasets, API credentials, collection state, or replace existing
experiment outputs without task authorization. Keep credentials out of logs and
commits. Avoid new network calls for offline preprocessing/training.

## Model progression
XGBoost -> independent-deck NN -> interaction-aware NN, with player-statistics
ablations and eventual separate self-/cross-attention ablations. XGBoost can learn
feature interactions too; the hypothesis concerns explicit neural interaction
modeling, not interactions versus none.
