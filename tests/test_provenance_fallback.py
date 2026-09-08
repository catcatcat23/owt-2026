import subprocess
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from main_pretrain_orgslot_common8_a100 import _git_provenance, _write_provenance


class ProvenanceFallbackTests(unittest.TestCase):
    def test_full_writer_without_git_still_records_sources_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'fixture.json'
            fixture.write_text('{}')
            args = SimpleNamespace(visibility_config=str(fixture),
                preprocess_summary=str(fixture), data_path=str(fixture),
                val_data_path=str(fixture), roi_index=None)
            with mock.patch('main_pretrain_orgslot_common8_a100.subprocess.run',
                            side_effect=FileNotFoundError('git unavailable')):
                _write_provenance(args, root / 'output')
            output = root / 'output'
            self.assertEqual((output / 'git_commit.txt').read_text().strip(), 'unavailable')
            hashes = json.loads((output / 'source_checksums.json').read_text())
            self.assertEqual(len(hashes), 5)
            self.assertTrue(all(len(v) == 64 for v in hashes.values()))
            self.assertTrue((output / 'dataset_manifest_checksums.json').is_file())

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
