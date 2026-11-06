# Frozen RoyaleAPI static catalogs

Snapshot requested on November 5, 2026. Source: https://github.com/RoyaleAPI/cr-api-data

Pinned commit: `d5461b0a59bff33c4da2fc845b07275b66b2d6ff` (October 18, 2023).

**Freshness:** This was the latest upstream master at download time, but it is not verified current game metadata. Do not describe these combat stats as the balance version for battles collected today. Missing/new cards must remain supported using raw battle card IDs; do not filter battles to this catalog.

- `cards.json`: 120 card entries with IDs, names, elixir, type, rarity and other metadata.
- `rarities.json`: rarity metadata and progression constants.
- `game_modes.json`: mode reference metadata.
- `cards_stats.json`: aggregate combat stats, with troop, building, spell, projectile, characters and character_buff sections.
- `cards_stats_*.json`: separate versions of those sections, retained as upstream files; do not concatenate them with the aggregate as independent records.
- `manifest.json`: immutable source URLs, commit, download timestamp and SHA-256 checksums.

These reference files are separate from the legacy `data/processed/card_list.json` and the preprocessing-generated observed `card_catalog.json`. Preprocessing does not automatically consume these files. Join metadata by card ID after validating coverage and source freshness. Keep the frozen files unchanged for reproducibility.
