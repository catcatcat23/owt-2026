import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.prepare_word_official96 import make_plan, check_plan


class OfficialSplitTest(unittest.TestCase):
    def test_frozen_plan_reproducible(self):
        plan = json.loads(Path("configs/orgslot/word_official96_seed42.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, ids in plan["official_pools"].items():
                (root / ("images" + split)).mkdir()
                for case in ids:
                    (root / ("images" + split) / (case + ".nii.gz")).touch()
            self.assertEqual(make_plan(root), plan)
        check_plan(plan)

    def test_no_overlap_and_unused_not_sampled(self):
        plan = json.loads(Path("configs/orgslot/word_official96_seed42.json").read_text())
        check_plan(plan)
        used = {e["case_id"] for rows in plan["splits"].values() for e in rows}
        self.assertEqual(len(used), 146)
        self.assertFalse(used & set(plan["unused_train"]))

    def test_reject_leak(self):
        plan = json.loads(Path("configs/orgslot/word_official96_seed42.json").read_text())
        broken = copy.deepcopy(plan)
        broken["splits"]["Test"][0] = broken["splits"]["Training"][0]
        with self.assertRaises(AssertionError):
            check_plan(broken)


if __name__ == "__main__":
    unittest.main()
