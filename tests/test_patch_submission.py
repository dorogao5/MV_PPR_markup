from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from piglabeler.patch_submission import patch_submission


class PatchSubmissionTests(unittest.TestCase):
    def test_patch_submission_applies_manual_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.csv"
            manual_path = root / "manual.csv"
            output_path = root / "patched.csv"

            with base_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["row_id", "class_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"row_id": "row_1", "class_id": 1},
                        {"row_id": "row_2", "class_id": 2},
                        {"row_id": "row_3", "class_id": 3},
                    ]
                )

            with manual_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["row_id", "source_name", "class_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"row_id": "row_2", "source_name": "base", "class_id": 4},
                        {"row_id": "row_3", "source_name": "other", "class_id": 0},
                        {"row_id": "row_missing", "source_name": "base", "class_id": 1},
                    ]
                )

            result = patch_submission(
                base_submission_path=base_path,
                manual_labels_path=manual_path,
                output_path=output_path,
                manual_source_name=None,
            )

            with output_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(
            rows,
            [
                {"row_id": "row_1", "class_id": "1"},
                {"row_id": "row_2", "class_id": "4"},
                {"row_id": "row_3", "class_id": "3"},
            ],
        )
        self.assertEqual(result.total_rows, 3)
        self.assertEqual(result.manual_rows_loaded, 2)
        self.assertEqual(result.manual_rows_applied, 1)
        self.assertEqual(result.manual_rows_missing_in_base, 1)


if __name__ == "__main__":
    unittest.main()
