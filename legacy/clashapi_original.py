"""
clashapi.py — BFS-based Clash Royale battle data collector.

Strategy:
  1. Seed from a list of clan tags and/or player tags.
  2. Fetch each player's battlelog (~25 most recent battles).
  3. Deduplicate battles by (battleTime, sorted team+opponent tags).
  4. Enqueue opponent tags discovered in each battle.
  5. Repeat until the queue is empty or the target battle count is reached.

State is persisted to disk so the run can be interrupted and resumed.
"""

import json
import os
import time
from collections import deque
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

BASE_URL = "https://proxy.royaleapi.dev/v1"
TOKEN = os.environ["CLASH_ROYALE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(exist_ok=True)

BATTLES_FILE = RAW_DIR / "battles.jsonl"
STATE_FILE = RAW_DIR / "collector_state.json"

# --- trophy range to keep data consistent ---
TROPHY_MIN = 3000
TROPHY_MAX = 9000

# --- rate limiting: stay under 10 req/s ---
REQUEST_DELAY = 0.15  # seconds between requests


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(url: str) -> dict | list:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_battlelog(player_tag: str) -> list:
    return _get(f"{BASE_URL}/players/{player_tag}/battlelog")


def get_player(player_tag: str) -> dict:
    return _get(f"{BASE_URL}/players/{player_tag}")


def get_clan_members(clan_tag: str) -> list[str]:
    """Return URL-encoded player tags for all clan members."""
    data = _get(f"{BASE_URL}/clans/{clan_tag}/members")
    return [
        m["tag"].replace("#", "%23")
        for m in data.get("items", [])
        if m.get("tag")
    ]


# ---------------------------------------------------------------------------
# State persistence (so the run is resumable)
# ---------------------------------------------------------------------------

def load_state() -> tuple[set, set, deque]:
    """Load seen_players, seen_battles, and the player queue from disk."""
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps({}))
        return set(), set(), deque()

    raw = json.loads(STATE_FILE.read_text())
    seen_players = set(raw.get("seen_players", []))
    seen_battles = set(raw.get("seen_battles", []))
    queue = deque(raw.get("queue", []))
    print(f"[resume] {len(seen_players)} players seen, "
          f"{len(seen_battles)} battles seen, {len(queue)} in queue")
    return seen_players, seen_battles, queue


def save_state(seen_players: set, seen_battles: set, queue: deque) -> None:
    STATE_FILE.write_text(json.dumps({
        "seen_players": list(seen_players),
        "seen_battles": list(seen_battles),
        "queue": list(queue),
    }))


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def battle_key(battle: dict) -> str | None:
    """Stable dedup key for a battle: battleTime + sorted player tags."""
    team = battle.get("team", [{}])[0].get("tag", "")
    opp = battle.get("opponent", [{}])[0].get("tag", "")
    t = battle.get("battleTime", "")
    if not (team and opp and t):
        return None
    return f"{t}|{'|'.join(sorted([team, opp]))}"


def is_1v1_ladder(battle: dict) -> bool:
    """Keep only standard 1v1 ladder battles."""
    game_mode = battle.get("gameMode", {}).get("name", "")
    team = battle.get("team", [])
    opp = battle.get("opponent", [])
    return len(team) == 1 and len(opp) == 1 and "Ladder" in game_mode


def in_trophy_range(battle: dict) -> bool:
    team_trophies = battle.get("team", [{}])[0].get("startingTrophies", 0) or 0
    return TROPHY_MIN <= team_trophies <= TROPHY_MAX


def extract_opponent_tag(battle: dict) -> str | None:
    opp_tag = battle.get("opponent", [{}])[0].get("tag", "")
    if opp_tag:
        return opp_tag.replace("#", "%23")
    return None


def collect(
    seed_clans: list[str],
    seed_players: list[str],
    target_battles: int = 10_000,
    state_save_interval: int = 50,
) -> None:
    seen_players, seen_battles, queue = load_state()
    if not BATTLES_FILE.exists():
        BATTLES_FILE.touch()
    total_written = sum(1 for _ in BATTLES_FILE.open())

    # Seed the queue on first run
    if not seen_players and not queue:
        print("[seed] Fetching clan members...")
        for clan_tag in seed_clans:
            try:
                members = get_clan_members(clan_tag)
                for tag in members:
                    if tag not in seen_players:
                        queue.append(tag)
                print(f"  {clan_tag}: {len(members)} members")
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                print(f"  [error] clan {clan_tag}: {e}")

        for tag in seed_players:
            encoded = tag.replace("#", "%23")
            if encoded not in seen_players:
                queue.append(encoded)

        print(f"[seed] Queue size: {len(queue)}")

    # BFS loop
    processed = 0
    while queue and total_written < target_battles:
        player_tag = queue.popleft()

        if player_tag in seen_players:
            continue
        seen_players.add(player_tag)

        try:
            battles = get_battlelog(player_tag)
            time.sleep(REQUEST_DELAY)
        except requests.HTTPError as e:
            if e.response.status_code in (404, 403):
                pass  # private/deleted account, skip silently
            else:
                print(f"[warn] {player_tag}: HTTP {e.response.status_code}")
            continue
        except Exception as e:
            print(f"[warn] {player_tag}: {e}")
            continue

        new_battles = 0
        with BATTLES_FILE.open("a", encoding="utf-8") as f:
            for battle in battles:
                if not is_1v1_ladder(battle) or not in_trophy_range(battle):
                    continue

                key = battle_key(battle)
                if key is None or key in seen_battles:
                    continue
                seen_battles.add(key)

                # Enqueue the opponent for BFS expansion
                opp_tag = extract_opponent_tag(battle)
                if opp_tag and opp_tag not in seen_players:
                    queue.append(opp_tag)

                f.write(json.dumps({"player_tag": player_tag, "battle": battle}) + "\n")
                new_battles += 1

        total_written += new_battles
        processed += 1

        print(f"[{processed}] {player_tag}: +{new_battles} battles | "
              f"total={total_written} | queue={len(queue)}")

        if processed % state_save_interval == 0:
            save_state(seen_players, seen_battles, queue)

    save_state(seen_players, seen_battles, queue)
    print(f"\nDone. Total battles on disk: {total_written}")


# ---------------------------------------------------------------------------
# Entry point — edit seeds here
# ---------------------------------------------------------------------------

SEED_CLANS = [
    "%23U2VJJG",   # your clan
    # add more clan tags here (URL-encoded, # → %23)
]

SEED_PLAYERS = [
    "%238928P288L",  # your tag
]

if __name__ == "__main__":
    collect(
        seed_clans=SEED_CLANS,
        seed_players=SEED_PLAYERS,
        target_battles=50_000,
    )
