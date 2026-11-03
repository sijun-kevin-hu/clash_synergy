# card_api.py

import requests
import json
from pathlib import Path

BASE_URL = "https://proxy.royaleapi.dev/v1/cards"
CARD_CACHE = Path("data/processed/card_list.json")
SUPPORT_CACHE = Path("data/processed/support_list.json")

def fetch_cards(api_token: str) -> list[str]:
    """
    Fetch the latest card list from the Clash Royale API and merge into
    the existing card_list.json, preserving manually assigned roles.
    New cards from the API are added with an empty role list.
    """

    # Load existing role mappings if present
    existing = {}
    if CARD_CACHE.exists():
        with open(CARD_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                existing = data
            else:
                # Legacy plain-list format: convert to dict with empty roles
                existing = {name: [] for name in data}

    url = BASE_URL
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # Merge new cards into existing mappings
    items = data.get("items", [])
    new_cards = []
    for card in items:
        name = card.get("name")
        if name and name not in existing:
            existing[name] = []
            new_cards.append(name)

    if new_cards:
        print(f"New cards added (assign roles in {CARD_CACHE}): {new_cards}")

    # Write back sorted dict preserving roles
    CARD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CARD_CACHE, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(existing.items())), f, indent=4)

    return sorted(existing.keys())

def fetch_support_cards(api_token: str) -> list[str]:
    """
    Fetch the latest support cards/items list from the Clash Royale API.
    Caches locally so you don't need to download every time.
    """
    
    if SUPPORT_CACHE.exists():
        with open(SUPPORT_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
        
    url = BASE_URL
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    # Extract support names from "supportItems"
    supportItems = data.get('supportItems', [])
    support_names = [support['name'] for support in supportItems if 'name' in support]
    support_names.sort()
    
    # Cache the result
    SUPPORT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUPPORT_CACHE, 'w', encoding='utf-8') as f:
        json.dump(support_names, f, indent=2)
        
    return support_names