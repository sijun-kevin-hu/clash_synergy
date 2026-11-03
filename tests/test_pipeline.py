import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from data_collection.clashapi import collect
from preprocessing.preprocessing import process

def battle(day, winner=True, mode='Ladder'):
    def player(tag, crowns):
        return {'tag': tag, 'startingTrophies': 6000, 'crowns': crowns,
                'cards': [{'id': i, 'level': 10, 'maxLevel': 14} for i in range(1, 9)]}
    return {'type': 'PvP', 'gameMode': {'name': mode}, 'battleTime': f'202601{day:02d}T120000.000Z',
            'team': [player('#AAA', int(winner))], 'opponent': [player('#BBB', int(not winner))]}

class PipelineTests(unittest.TestCase):
    def test_filter_history_duplicates_and_chronology(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'raw.jsonl'
            records = [battle(i) for i in range(1, 11)]
            records += [battle(1), battle(11, mode='Rage_Ladder')]
            tie = battle(12); tie['opponent'][0]['crowns'] = 1; records.append(tie)
            broken = battle(13); broken['team'][0]['cards'].pop(); records.append(broken)
            raw.write_text(''.join(json.dumps({'battle': b})+'\n' for b in records))
            report = process(raw, Path(tmp)/'out')
            self.assertEqual(report['usable_matches'], 10)
            self.assertEqual(report['counts']['duplicate'], 1)
            rows = [json.loads(s) for s in (Path(tmp)/'out/matches.jsonl').read_text().splitlines()]
            for i, r in enumerate(rows):
                self.assertEqual(r['p1']['prior_observed_games'], i)
                self.assertEqual(r['p2']['prior_observed_games'], i)
                self.assertEqual(r['label'], int(r['p1']['player_tag'] == '#AAA'))
            self.assertLess(max(r['battle_time'] for r in rows if r['split']=='train'), min(r['battle_time'] for r in rows if r['split']=='validation'))

    def test_equal_timestamps_do_not_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'raw.jsonl'; a = battle(1); b = copy.deepcopy(a)
            b['opponent'][0]['tag'] = '#CCC'
            raw.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
            process(raw, Path(tmp)/'out')
            rows = [json.loads(s) for s in (Path(tmp)/'out/matches.jsonl').read_text().splitlines()]
            self.assertEqual({r['split'] for r in rows}, {'test'})
            self.assertTrue(all(r[s]['prior_observed_games']==0 for r in rows for s in ['p1','p2']))

    def test_collector_additional_target_and_resume(self):
        class Response:
            status_code=200
            def raise_for_status(self): pass
            def json(self): return [battle(i) for i in range(1, 6)]
        class Session:
            def __init__(self): self.headers={}
            def get(self, *args, **kwargs): return Response()
            def close(self): pass
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(raw_dir=tmp, delay=0, base_url='https://example.invalid', seed_clan=[], seed_player=['#AAA'], revisit=False, target_battles=2, min_trophies=0, max_trophies=100000)
            with patch.dict('os.environ', {'CLASH_ROYALE_API_TOKEN':'test'}), patch('data_collection.clashapi.requests.Session', Session):
                collect(args); collect(args)
            rows = [json.loads(s) for s in (Path(tmp)/'battles.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({r['battle']['battleTime'] for r in rows}),4)

if __name__ == '__main__': unittest.main()
