"""Offline preprocessing: complete decks, chronological splits and strictly prior history."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from utils.battles import battle_key, is_trophy_road

ROOT = Path(__file__).resolve().parents[1]

def timestamp(s):
    return datetime.strptime(s, '%Y%m%dT%H%M%S.%fZ').replace(tzinfo=timezone.utc)

def valid_number(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)

def validate(b):
    if not is_trophy_road(b): return 'wrong_mode'
    try: timestamp(b['battleTime'])
    except (ValueError, KeyError, TypeError): return 'missing_or_invalid_timestamp'
    if not battle_key(b): return 'missing_player_ids'
    if b['team'][0]['tag'] == b['opponent'][0]['tag']: return 'same_player'
    for p in [b['team'][0], b['opponent'][0]]:
        cards = p.get('cards', [])
        if len(cards) != 8 or any(not isinstance(c, dict) for c in cards): return 'incomplete_deck'
        if any(not isinstance(c.get('id'), int) or not valid_number(c.get('level')) or not valid_number(c.get('maxLevel')) or c['level'] <= 0 or c['maxLevel'] <= 0 for c in cards): return 'invalid_card_fields'
        if len({c['id'] for c in cards}) != 8: return 'duplicate_card_in_deck'
        if not valid_number(p.get('crowns')) or p['crowns'] < 0: return 'missing_outcome'
        if not valid_number(p.get('startingTrophies')) or p['startingTrophies'] < 0: return 'missing_trophies'
    if b['team'][0]['crowns'] == b['opponent'][0]['crowns']: return 'tie'
    return None

def process(raw_path, output, seed=42, train_fraction=.8, val_fraction=.1):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    counts = Counter(); seen = set(); matches = []
    with Path(raw_path).open() as f:
        for line in f:
            if not line.strip(): continue
            counts['raw_records'] += 1
            try:
                record = json.loads(line); b = record.get('battle', record)
                if not isinstance(b, dict): raise ValueError('not a battle')
                reason = validate(b)
            except (ValueError, TypeError, KeyError, AttributeError):
                counts['malformed_record'] += 1; continue
            if reason: counts[reason] += 1; continue
            key = battle_key(b)
            if key in seen: counts['duplicate'] += 1; continue
            seen.add(key); matches.append((timestamp(b['battleTime']), key, b))
    matches.sort(key=lambda x: (x[0], x[1]))
    if not matches: raise ValueError('No usable matches; inspect raw data and filters.')
    # Boundaries are timestamps: simultaneous matches always belong to the same split.
    n = len(matches)
    train_end = matches[min(n-1, int(n * train_fraction))][0]
    val_end = matches[min(n-1, int(n * (train_fraction + val_fraction)))][0]
    histories = defaultdict(lambda: [0, 0])
    catalog = {}; features = []; split_counts = Counter(); labeled = Counter()
    with (output / 'matches.jsonl').open('w') as neural:
        i = 0
        while i < n:
            j = i + 1
            while j < n and matches[j][0] == matches[i][0]: j += 1
            updates = []
            for dt, key, b in matches[i:j]:
                a, z = b['team'][0], b['opponent'][0]
                if hashlib.sha256(f'{seed}|{key}'.encode()).digest()[0] < 128: a, z = z, a
                label = int(a['crowns'] > z['crowns'])
                split = 'train' if dt < train_end else 'validation' if dt < val_end else 'test'
                row = {'match_id': key, 'battle_time': dt.isoformat(), 'split': split, 'label': label}
                flat = dict(row)
                for side, p in [('p1', a), ('p2', z)]:
                    games, wins = histories[p['tag']]
                    cards = sorted(p['cards'], key=lambda c: c['id'])
                    payload = {
                        'player_tag': p['tag'], 'starting_trophies': p['startingTrophies'],
                        'prior_observed_games': games,
                        'prior_observed_win_rate': wins / games if games else None,
                        'cards': [{k: c[k] for k in ('id', 'name', 'level', 'maxLevel', 'evolutionLevel', 'elixirCost') if k in c} for c in cards],
                        'support_cards': p.get('supportCards', []),
                    }
                    row[side] = payload
                    for field in ('player_tag', 'starting_trophies', 'prior_observed_games', 'prior_observed_win_rate'): flat[f'{side}_{field}'] = payload[field]
                    for c in cards:
                        catalog[str(c['id'])] = c.get('name', str(c['id']))
                        flat[f'{side}_card_{c["id"]}_level'] = c['level'] / c['maxLevel']
                        # Missing evolution information stays missing, rather than inventing a zero.
                        flat[f'{side}_card_{c["id"]}_evolution'] = c.get('evolutionLevel', '')
                    # Tower troops may affect outcomes; include when recorded.
                    for c in p.get('supportCards', []):
                        if c.get('id') and valid_number(c.get('level')) and valid_number(c.get('maxLevel')) and c['maxLevel'] > 0:
                            flat[f'{side}_support_{c["id"]}_level'] = c['level'] / c['maxLevel']
                flat['trophy_diff'] = a['startingTrophies'] - z['startingTrophies']
                neural.write(json.dumps(row) + '\n'); features.append(flat)
                split_counts[split] += 1; labeled[label] += 1
                updates.extend([(a['tag'], label), (z['tag'], 1-label)])
            # No same-time results can enter another match's history.
            for player, won in updates:
                histories[player][0] += 1; histories[player][1] += won
            i = j
    columns = sorted({k for row in features for k in row})
    with (output / 'baseline.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in features:
            # An absent card is zero; an unrecorded evolution level remains blank.
            writer.writerow({c: row.get(c, 0 if '_card_' in c or '_support_' in c else '') for c in columns})
    (output / 'card_catalog.json').write_text(json.dumps(catalog, indent=2, sort_keys=True))
    report = {'counts': dict(counts), 'usable_matches': n, 'splits': dict(split_counts), 'labels': dict(labeled),
              'first_battle': matches[0][0].isoformat(), 'last_battle': matches[-1][0].isoformat(),
              'train_end_exclusive': train_end.isoformat(), 'validation_end_exclusive': val_end.isoformat(),
              'unique_cards': len(catalog), 'unique_players': len(histories), 'seed': seed,
              'history_policy': 'Only strictly earlier observed matches; histories update during validation/test (rolling evaluation).'}
    (output / 'quality_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=str(ROOT / 'data/raw/battles.jsonl'))
    p.add_argument('--output-dir', default=str(ROOT / 'data/processed'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-fraction', type=float, default=.8)
    p.add_argument('--validation-fraction', type=float, default=.1)
    a = p.parse_args()
    if not 0 < a.train_fraction < 1 or not 0 < a.validation_fraction < 1 or a.train_fraction + a.validation_fraction >= 1: p.error('Fractions must be positive and sum to less than 1')
    process(a.input, a.output_dir, a.seed, a.train_fraction, a.validation_fraction)

if __name__ == '__main__': main()
