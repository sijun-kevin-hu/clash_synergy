"""Resumable opponent crawl; collect up to 250,000 NEW Trophy Road battles per invocation."""
import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote, unquote
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env.local')

from utils.battles import tag, battle_key, is_trophy_road
from data_collection.dashboard import Dashboard

def collect(args):
    token = os.environ.get('CLASH_ROYALE_API_TOKEN')
    if not token:
        raise SystemExit('Set CLASH_ROYALE_API_TOKEN in .env.local or the environment.')
    raw_dir = Path(args.raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    battles_file, state_file = raw_dir / 'battles.jsonl', raw_dir / 'collector_state.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    queue = deque(tag(t) for t in state.get('queue', []))
    queued = set(queue)
    seen_players = {tag(t) for t in state.get('seen_players', [])}
    # Raw records are the source of truth after interruption or an older state file.
    seen_battles = set()
    if battles_file.exists():
        with battles_file.open() as f:
            for i, line in enumerate(f, 1):
                try:
                    key = battle_key(json.loads(line)['battle'])
                except (ValueError, KeyError, TypeError):
                    raise SystemExit(f'Malformed raw record on line {i}; repair it before resuming.')
                if key: seen_battles.add(key)
    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {token}'})
    dashboard = Dashboard(args.target_battles, getattr(args, 'ui', 'auto'))
    dashboard.update(existing=len(seen_battles), queued=len(queue))
    def get(path):
        for attempt in range(6):
            time.sleep(args.delay)
            try:
                dashboard.update(requests=dashboard.requests + 1, status='Polling the arena')
                response = session.get(args.base_url.rstrip('/') + path, timeout=30)
                if response.status_code in (401, 403):
                    raise SystemExit('API authorization failed: check token, endpoint and IP configuration.')
                if response.status_code == 429 or response.status_code >= 500:
                    wait = min(60, 2 ** attempt)
                    dashboard.update(retries=dashboard.retries + 1, status=f'HTTP {response.status_code} · backing off {wait}s')
                    dashboard.event(f'HTTP {response.status_code} · retry {attempt + 1}/6')
                    time.sleep(wait); continue
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 5: raise
                wait = min(60, 2 ** attempt)
                dashboard.update(retries=dashboard.retries + 1, status=f'Connection interrupted · backing off {wait}s')
                dashboard.event(f'Connection retry {attempt + 1}/6')
                time.sleep(wait)
        raise requests.RequestException('Retry budget exhausted')
    def enqueue(t, force=False):
        t = tag(t)
        if t not in queued and (force or t not in seen_players):
            queued.add(t); queue.append(t)
    def save():
        tmp = state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps({'seen_players': sorted(seen_players), 'queue': list(queue)}))
        tmp.replace(state_file)
    added = processed = 0
    try:
        dashboard.start()
        dashboard.event(f'Resumed archive · {len(seen_battles):,} unique battles')
        for clan in args.seed_clan:
            for member in get('/clans/' + quote(tag(clan), safe='') + '/members').get('items', []):
                enqueue(member['tag'])
        for player in args.seed_player: enqueue(player, force=args.revisit)
        if not queue:
            raise SystemExit('No queued players. Supply --seed-player or --seed-clan; --revisit refreshes old players.')
        with battles_file.open('a', encoding='utf-8') as out:
            while queue and added < args.target_battles:
                player = queue.popleft(); queued.discard(player)
                if player in seen_players and not args.revisit: continue
                dashboard.update(player=player, queued=len(queue))
                before = added
                # Preserve the current player in state until its complete log is handled.
                try:
                    logs = get('/players/' + quote(player, safe='') + '/battlelog')
                except requests.HTTPError as e:
                    if e.response.status_code == 404:
                        dashboard.event(f'{player} · player not found')
                        seen_players.add(player); continue
                    queue.appendleft(player); raise
                except BaseException:
                    queue.appendleft(player); raise
                try:
                    hit_limit = False
                    for b in logs:
                        # Explore all 1v1 opponents, even when their battle isn't eligible.
                        if len(b.get('opponent', [])) == 1 and b['opponent'][0].get('tag'):
                            enqueue(b['opponent'][0]['tag'])
                        if not is_trophy_road(b):
                            dashboard.filtered += 1
                            continue
                        trophies = [p.get('startingTrophies') for p in [b['team'][0], b['opponent'][0]]]
                        if any(not isinstance(t, (int, float)) or t < args.min_trophies or t > args.max_trophies for t in trophies):
                            dashboard.filtered += 1
                            continue
                        key = battle_key(b)
                        if not key:
                            dashboard.filtered += 1
                            continue
                        if key in seen_battles:
                            dashboard.duplicates += 1
                            continue
                        out.write(json.dumps({'player_tag': player, 'battle': b}) + '\n')
                        seen_battles.add(key); added += 1
                        if added >= args.target_battles:
                            hit_limit = True; break
                    out.flush()
                except BaseException:
                    queue.appendleft(player); raise
                if hit_limit:
                    queue.appendleft(player)
                else:
                    seen_players.add(player)
                processed += 1
                dashboard.update(added=added, processed=processed, queued=len(queue), status='Exploring opponent frontier')
                dashboard.event(f'{player} · +{added - before} battles · {len(logs)} scanned')
                if processed % 50 == 0:
                    save()
                    dashboard.event('Checkpoint saved')
                    if not dashboard.enabled:
                        print(f'players={processed} new_battles={added} queue={len(queue)}', flush=True)
    finally:
        try:
            save()
        finally:
            session.close()
            dashboard.close()
        print(f'Added {added:,} new unique battles (requested {args.target_battles:,}).')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-battles', type=int, default=250_000, help='Additional unique battles this invocation')
    p.add_argument('--raw-dir', default=str(ROOT / 'data/raw'))
    p.add_argument('--base-url', default=os.environ.get('CLASH_ROYALE_BASE_URL', 'https://proxy.royaleapi.dev/v1'))
    p.add_argument('--seed-clan', action='append', default=[])
    p.add_argument('--seed-player', action='append', default=[])
    p.add_argument('--min-trophies', type=int, default=0)
    p.add_argument('--max-trophies', type=int, default=100_000)
    p.add_argument('--delay', type=float, default=0.15)
    p.add_argument('--revisit', action='store_true', help='Requeue previously visited players to fetch newer logs')
    p.add_argument('--ui', choices=['auto', 'plain'], default='auto', help='Live terminal dashboard (auto) or plain log output')
    args = p.parse_args()
    if args.target_battles < 1 or args.delay < 0 or args.min_trophies > args.max_trophies: p.error('Invalid target, delay or trophy range')
    if args.revisit:
        state_path = Path(args.raw_dir) / 'collector_state.json'
        if state_path.exists():
            args.seed_player.extend(json.loads(state_path.read_text()).get('seen_players', []))
            # Allow enqueueing old players during this invocation without losing the saved state.
            # collect uses seen_players to suppress duplicates; explicit refresh is handled below.
    try:
        collect(args)
    except KeyboardInterrupt:
        print('Stopped. Progress saved; run the same command to resume.')

if __name__ == '__main__': main()
