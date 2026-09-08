import subprocess
import unittest
from unittest import mock

from main_pretrain_orgslot_common8_a100 import _git_provenance


class ProvenanceFallbackTests(unittest.TestCase):
    def test_non_git_snapshot_is_nonfatal_and_explicit(self):
        failure = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        with mock.patch("main_pretrain_orgslot_common8_a100.subprocess.run",
                        side_effect=failure):
            values, errors = _git_provenance()
        self.assertEqual(values, {"commit": "", "status": "", "diff": ""})
        self.assertEqual(set(errors), {"commit", "status", "diff"})
        self.assertTrue(all("CalledProcessError" in value for value in errors.values()))

    def test_git_metadata_is_preserved_when_available(self):
        completed = subprocess.CompletedProcess(["git"], 0, stdout="value\n")
        with mock.patch("main_pretrain_orgslot_common8_a100.subprocess.run",
                        return_value=completed):
            values, errors = _git_provenance()
        self.assertEqual(values, {
            "commit": "value\n", "status": "value\n", "diff": "value\n"
        })
        self.assertEqual(errors, {})
