# ClashSynergy data-fit report

Audit date: November 3, 2026.

## Verdict

The existing raw battle data supports the proposed deck-input representation, outcome labels, chronological splits and a restricted historical-statistics experiment. It does not yet satisfy the planned sample size, establish reliable lifetime player skill, or demonstrate current-game coverage. The new preprocessing closes the structural gaps in the old feature CSV; collect more data before making claims about generalization or synergy coverage.

Sources inspected: the team's `CS4644_Project.pdf` (project definition, especially pages 1–3), original Clash Predict collection/preprocessing code, and the entire `/Users/kevinhu/Programming/clash_predict/data/raw/battles.jsonl`. This is an audit of local historical records, not a live API validation. The separate assignment-instructions PDF was not treated as the team's proposal.

## Measured results

| Check | Result |
|---|---:|
| Raw records | 50,010 |
| Duplicate battle keys across all raw modes | 164 |
| Non–Trophy Road records | 726 |
| Duplicate eligible records removed after mode filtering | 159 |
| Tied eligible records excluded | 3 |
| Usable unique binary Trophy Road matches | 49,122 |
| Train / validation / test | 39,297 / 4,912 / 4,913 |
| Player 1 wins / losses after randomized assignment | 24,515 / 24,607 |
| Unique observed card IDs in usable data | 121 |
| Unique observed players in usable data | 46,850 |
| Date range of usable battles, UTC | January 1–March 15, 2026 |

Filtering counts are sequential, so 159 eligible duplicates differs from the 164 duplicate keys across all raw modes. After these exclusions, no additional records were rejected for incomplete decks, invalid essential card fields, missing trophies or missing outcomes. Both players' tags, starting trophies, eight-card decks, card IDs/levels/max levels and crowns were present on every raw 1v1 player side inspected. The original broad substring test admitted clan-war modes such as Rage_Ladder and TripleElixir_Ladder; the new collector requires exact PvP/Ladder.

## Fit against the proposal

| Proposed need | Fit and action |
|---|---|
| Both eight-card decks and card levels | Available; new JSONL retains raw IDs and levels for embeddings and set/attention models. |
| Pre-match information only | Starting trophies and recorded decks are available. Crowns form the label only. Do not attach current-profile statistics to past battles. |
| Winner labels | Available; ties excluded rather than called losses. |
| Deduplication | Implemented using battle timestamp plus sorted player IDs, independently of player orientation. |
| Random competitor assignment | Implemented reproducibly; corresponding winner label follows the swap. |
| Chronological train/validation/test split | Implemented with equal timestamps kept together; strict split boundaries verified on existing data. |
| Historical player statistics | Only prior observed game count and win rate are computed. These are incomplete sampled histories, not lifetime records. |
| At least 50,000 usable matches | Existing data falls short by 878; new target is 250,000 additional raw qualifying unique matches per invocation, not guaranteed 250,000 training-ready matches. |
| Fair model and ablation comparisons | Data structure supports them; model training, shared feature policies and evaluation code remain to be implemented. |
| Season 18 dataset comparison | Not imported or audited here. Keep its earlier game environment separate unless schema and domain shifts are explicitly addressed. |

## Material limitations and remaining decisions

1. **Player history is sparse.** Among 98,244 usable player appearances, 46,850 (47.7%) have no earlier observed match and 58,482 (59.5%) have fewer than five. Collecting repeated observations over time is more useful for this experiment than only expanding to new players. Null win rates and prior-game counts expose uncertainty. Report results separately for players with sufficient history. Rolling evaluation incorporates earlier validation/test outcomes only after their battle time; use a different explicitly documented history policy if the experiment requires a fixed deployment snapshot.

2. **Evolution and tower configuration need a shared policy.** In usable decks, 658,796 of 785,952 card entries lack an explicit evolutionLevel field (83.8%). Absence alone has not been validated as proof of non-evolution, so missing values remain missing. There are 2,905 missing elixirCost fields (0.37%). Tower/support arrays are present in the inspected raw data and are preserved, but their semantic completeness was not independently verified. Additional mechanics represented by optional API fields may need future preprocessing support. For all three models, either consistently encode verified fields or document controlled exclusions; do not let only one model receive extra information.

3. **The data is historical and temporally concentrated.** It spans January–March 2026, while the train split ends March 14 and validation ends March 15. Most matches therefore cluster near collection time; the test interval is short. More collection over multiple days/weeks and game balance periods would strengthen temporal evaluation. The old collection used a 3,000–9,000 trophy filter on the queried player's side, which biases coverage. The new collector removes that narrow restriction and applies configurable limits symmetrically.

4. **Coverage is not established by row count.** Seeing 121 card IDs does not prove all current cards or rare pairs are covered. Before model conclusions, inspect card/pair frequency, train-only vocabulary versus unseen test cards, repeated-deck frequency, player overlap and performance on rare/unseen deck combinations. The supplied report does not claim those analyses are complete.

5. **Sampling can bias outcomes.** Opponent BFS is a connected convenience sample rather than a representative sample of all Trophy Road matches. Use diverse seed clans/players and report trophy and date distributions. Player overlap across chronological splits is expected for a rolling experiment; a separate player-disjoint test answers a different generalization question.

6. **Runtime and operational limits.** The collector can stop below target when its queue empties, on authorization failure, or after exhausting retries. It retries rate-limit/server/transient network failures and fails clearly on authorization errors. Current endpoint credentials, IP configuration and mode identifiers have not been live-tested. Run a small collection before scaling. Preprocessing currently keeps matches and baseline rows in memory; 250,000–multiple-run datasets may require substantial RAM and eventually streaming/columnar output. Run only one collector per raw directory.

## Changes made

- Copied original scripts into `legacy/`, card helper into `utils/`, and historical card/support metadata without credentials, raw datasets or model artifacts.
- Added configurable collection with 250,000 additional matches per run, exact mode filtering, strict target enforcement, retry handling, raw-file-based restart deduplication, atomic state and refresh support.
- Replaced the main preprocessing entry point with offline model-ready match records, baseline CSV, timestamp/player metadata, deterministic side assignment, tie/incomplete-data filtering, chronological splits and strictly prior observed histories.
- Kept original raw card levels and max levels; normalized ratios are provided only in the baseline table. Slot order must not be interpreted as a sequence.
- Added setup instructions, environment template, ignore rules and targeted regression tests.

## Verification

The complete historical raw file was processed successfully; all three regression tests passed. Tests cover collector additional-count semantics and resume without duplicate writes, preprocessing exclusions, orientation-consistent labels, temporal separation and same-timestamp history isolation. No live requests, new collection run, training or predictive-quality claims were performed.

Machine-readable preprocessing results are in `reports/existing_data_quality.json`. The old raw dataset remains in the previous project; pass its path through `--input` to regenerate processed data locally.
