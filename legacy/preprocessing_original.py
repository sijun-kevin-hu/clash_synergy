# preprocess_cards.py

import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from utils.fetch_cards import fetch_cards, fetch_support_cards

load_dotenv(".env.local")

RAW_PATH = Path("data/raw/battles.jsonl")
OUT_PATH = Path("data/processed/battles_cards.csv")
CARD_CACHE = Path("data/processed/card_list.json")

# Adjust MIN MAX trophy range
MIN_TROPHY = 3000
MAX_TROPHY = 9000

ROLE_CATEGORIES = [
    "win condition", "spell", "air", "mini-tank",
    "building", "swarm", "cycle", "support",
]

def load_card_list():
    return fetch_cards(os.environ["CLASH_ROYALE_API_TOKEN"])

def load_support_list():
    return fetch_support_cards(os.environ["CLASH_ROYALE_API_TOKEN"])

def load_card_roles():
    """Load the card name -> roles mapping from card_list.json."""
    if CARD_CACHE.exists():
        with open(CARD_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def create_deck_balance_features(deck_cards, card_roles, prefix):
    """
    Count how many cards in the deck belong to each role category.
    Cards with multiple roles count toward each.
    """
    role_counts = {role: 0 for role in ROLE_CATEGORIES}
    for card in deck_cards:
        name = card.get("name")
        for role in card_roles.get(name, []):
            if role in role_counts:
                role_counts[role] += 1

    return {
        f"{prefix}_{role.replace(' ', '_')}_count": count
        for role, count in role_counts.items()
    }

def create_card_feature(deck_cards, card_list, prefix):
    """
    Create one-hot presence flags for a deck.
    """
    row = {}
    card_norm_level_map = {c.get("name"): c.get("level", 0) / c.get("maxLevel", 1) for c in deck_cards}
    for card in card_list:
        key = f"{prefix}_norm_level_{card.replace(' ', '_')}"
        row[key] = card_norm_level_map[card] if card in card_norm_level_map else 0
    return row

def create_support_card_feature(deck_cards, support_list, prefix):
    row = {}
    support_norm_level_map = {c.get("name"): c.get("level", 0) / c.get("maxLevel", 1) for c in deck_cards}
    for support in support_list:
        key = f"{prefix}_support_norm_level_{support.replace(' ', '_')}"
        row[key] = support_norm_level_map[support] if support in support_norm_level_map else 0
    return row


def preprocess_battle(record, card_list, support_list, card_roles):
    battle = record["battle"]

    # You can include your existing filtering (e.g., 1v1 ladder)
    team_list = battle.get("team", [])
    opp_list = battle.get("opponent", [])
    if len(team_list) != 1 or len(opp_list) != 1:
        return None
    
    # PvP Only
    if battle.get("type") != "PvP":
        return None

    # Trophy Road Only
    game_mode = battle.get("gameMode", {}).get("name", "")
    if game_mode != "Ladder":
        return None

    team = team_list[0]
    opp = opp_list[0]

    team_cards = team.get("cards", [])
    opp_cards = opp.get("cards", [])
    
    # Trophy Range Limit
    team_trophies = team.get("startingTrophies", 0)
    if team_trophies < MIN_TROPHY or team_trophies > MAX_TROPHY:
        return None
    

    # Basic numeric features
    team_trophies = team.get("startingTrophies") or 0
    opp_trophies = opp.get("startingTrophies") or 0
    row = {
        "team_trophies": team_trophies,
        "opp_trophies": opp_trophies,
        "trophy_diff": team_trophies - opp_trophies,
        "team_avg_elixir": sum(c.get("elixirCost", 0) for c in team_cards) / len(team_cards) if team_cards else 0,
        "opp_avg_elixir": sum(c.get("elixirCost", 0) for c in opp_cards) / len(opp_cards) if opp_cards else 0,
    }

    # Level-based card features
    row.update(create_card_feature(team_cards, card_list, "team"))
    row.update(create_card_feature(opp_cards, card_list, "opp"))

    # Support tower features
    team_support = team.get("supportCards", [])
    opp_support = opp.get("supportCards", [])
    row.update(create_support_card_feature(team_support, support_list, "team"))
    row.update(create_support_card_feature(opp_support, support_list, "opp"))

    # Deck balance features
    row.update(create_deck_balance_features(team_cards, card_roles, "team"))
    row.update(create_deck_balance_features(opp_cards, card_roles, "opp"))
    
    # Optional interaction features
    # for card in card_list:
    #     team_key = f"team_has_{card.replace(' ', '_')}"
    #     opp_key = f"opp_has_{card.replace(' ', '_')}"
    #     row[f"{card.replace(' ', '_')}_diff"] = row.get(team_key, 0) - row.get(opp_key, 0)
    #     row[f"{card.replace(' ', '_')}_both"] = row.get(team_key, 0) * row.get(opp_key, 0)

    # Label win/loss
    team_crowns = team.get("crowns", 0)
    opp_crowns = opp.get("crowns", 0)
    row["label"] = 1 if team_crowns > opp_crowns else 0

    return row

def preprocess_matchup(team_player, opp_player, card_list, support_list, card_roles):
    """
    Build a single-row DataFrame for a live matchup prediction.
    Accepts two player API responses (from /players endpoint).
    """
    team_cards = team_player.get("currentDeck", [])
    opp_cards = opp_player.get("currentDeck", [])

    team_trophies = team_player.get("trophies", 0)
    opp_trophies = opp_player.get("trophies", 0)

    row = {
        "team_trophies": team_trophies,
        "opp_trophies": opp_trophies,
        "trophy_diff": team_trophies - opp_trophies,
        "team_avg_elixir": sum(c.get("elixirCost", 0) for c in team_cards) / len(team_cards) if team_cards else 0,
        "opp_avg_elixir": sum(c.get("elixirCost", 0) for c in opp_cards) / len(opp_cards) if opp_cards else 0,
    }

    # Level-based card features
    row.update(create_card_feature(team_cards, card_list, "team"))
    row.update(create_card_feature(opp_cards, card_list, "opp"))

    # Support tower features
    team_support = team_player.get("currentDeckSupportCards", [])
    opp_support = opp_player.get("currentDeckSupportCards", [])
    row.update(create_support_card_feature(team_support, support_list, "team"))
    row.update(create_support_card_feature(opp_support, support_list, "opp"))

    # Deck balance features
    row.update(create_deck_balance_features(team_cards, card_roles, "team"))
    row.update(create_deck_balance_features(opp_cards, card_roles, "opp"))

    return pd.DataFrame([row])


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    card_list = load_card_list()
    card_list = sorted(card_list)

    support_list = load_support_list()
    support_list = sorted(support_list)

    card_roles = load_card_roles()

    rows = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            row = preprocess_battle(record, card_list, support_list=support_list, card_roles=card_roles)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    print("Processed rows:", len(df))
    df.to_csv(OUT_PATH, index=False)
    print("Saved enhanced dataset with card presence to", OUT_PATH)

if __name__ == "__main__":
    main()
