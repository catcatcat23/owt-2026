import json
from pathlib import Path
import tempfile
import unittest

from tools.prepare_word_official96 import make_plan, check_plan


class SplitSeedsTests(unittest.TestCase):
    def test_reproducible_plans_and_unchanged_seed42(self):
        reference = json.loads(Path('configs/orgslot/word_official96_seed42.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for split, cases in reference['official_pools'].items():
                folder = raw / ('images' + split)
                folder.mkdir()
                for case in cases:
                    (folder / (case + '.nii.gz')).touch()
            self.assertEqual(make_plan(raw), reference)
            tests = []
            for seed in (0, 1, 42):
                plan = make_plan(raw, seed)
                check_plan(plan)
                self.assertEqual(plan, json.loads(Path('configs/orgslot/word_official96_seed%d.json' % seed).read_text()))
                tests.append({c['case_id'] for c in plan['splits']['Test']})
            self.assertNotEqual(tests[0], tests[1])
            self.assertNotEqual(tests[0], tests[2])
            self.assertNotEqual(tests[1], tests[2])
