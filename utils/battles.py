from urllib.parse import unquote

def tag(value):
    return '#' + unquote(value).strip().lstrip('#').upper()

def battle_key(b):
    a, z = b.get('team', []), b.get('opponent', [])
    if len(a) != 1 or len(z) != 1 or not b.get('battleTime') or not a[0].get('tag') or not z[0].get('tag'):
        return None
    return '|'.join([b['battleTime'], *sorted([tag(a[0]['tag']), tag(z[0]['tag'])])])

def is_trophy_road(b):
    return b.get('type') == 'PvP' and b.get('gameMode', {}).get('name') == 'Ladder' and len(b.get('team', [])) == len(b.get('opponent', [])) == 1

