import unittest

from tools.split_orgslot_manifest import split_records


def _record(case_id, slice_index):
    return {
        "image_pth": f"/image/{case_id}/{case_id}_{slice_index}.jpg",
        "mask_pth": f"/mask/{case_id}/{case_id}_{slice_index}.png",
    }


class SplitManifestTests(unittest.TestCase):
    def test_split_is_deterministic_and_case_disjoint(self):
        records = [
            _record(f"case_{case}", slice_index)
            for case in range(5)
            for slice_index in range(3)
        ]
        first = split_records(records, validation_fraction=0.4, seed=7)
        second = split_records(records, validation_fraction=0.4, seed=7)
        self.assertEqual(first, second)
        train, validation, train_cases, validation_cases = first
        self.assertFalse(set(train_cases) & set(validation_cases))
        self.assertEqual(len(train) + len(validation), len(records))
        self.assertEqual(len(validation_cases), 2)

    def test_single_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two cases"):
            split_records([_record("one", 0)])


if __name__ == "__main__":
    unittest.main()
