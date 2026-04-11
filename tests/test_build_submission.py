from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from piglabeler.build_submission import build_submission


class BuildSubmissionTests(unittest.TestCase):
    def test_build_submission_applies_manual_overrides_from_current_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "test.csv"
            probs_dir = root / "probs"
            output_path = root / "submissions" / "submission.csv"
            manual_labels_path = root / "current_annotations.csv"
            probs_dir.mkdir()

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["row_id", "image_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"row_id": "row_1", "image_id": "frame_1.jpg"},
                        {"row_id": "row_2", "image_id": "frame_2.jpg"},
                        {"row_id": "row_3", "image_id": "frame_3.jpg"},
                    ]
                )

            effnet = np.asarray(
                [
                    [0.05, 0.80, 0.05, 0.05, 0.05],
                    [0.05, 0.10, 0.75, 0.05, 0.05],
                    [0.05, 0.10, 0.10, 0.70, 0.05],
                ]
            )
            maxvit = np.asarray(
                [
                    [0.10, 0.70, 0.10, 0.05, 0.05],
                    [0.05, 0.10, 0.65, 0.10, 0.10],
                    [0.10, 0.10, 0.10, 0.65, 0.05],
                ]
            )
            swin = np.asarray(
                [
                    [0.10, 0.75, 0.05, 0.05, 0.05],
                    [0.10, 0.10, 0.60, 0.15, 0.05],
                    [0.05, 0.15, 0.05, 0.70, 0.05],
                ]
            )

            np.save(probs_dir / "probs_t1_effnet_tta.npy", effnet)
            np.save(probs_dir / "probs_t1_maxvit_tta.npy", maxvit)
            np.save(probs_dir / "probs_t1_swin_tta.npy", swin)

            with manual_labels_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["row_id", "source_name", "class_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"row_id": "row_2", "source_name": "test", "class_id": 4},
                        {"row_id": "row_missing", "source_name": "test", "class_id": 1},
                        {"row_id": "row_1", "source_name": "train", "class_id": 3},
                    ]
                )

            result = build_submission(
                csv_path=csv_path,
                output_path=output_path,
                probs_dir=probs_dir,
                strategy="ensemble-mean",
                model_name="swin_tta",
                raw_weights=[],
                manual_labels_path=manual_labels_path,
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

    def test_build_submission_supports_weighted_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "test.csv"
            probs_dir = root / "probs"
            output_path = root / "submission.csv"
            probs_dir.mkdir()

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["row_id", "image_id"])
                writer.writeheader()
                writer.writerows([{"row_id": "row_1", "image_id": "frame_1.jpg"}])

            np.save(probs_dir / "probs_t1_effnet_tta.npy", np.asarray([[0.10, 0.80, 0.10, 0.00, 0.00]]))
            np.save(probs_dir / "probs_t1_maxvit_tta.npy", np.asarray([[0.10, 0.15, 0.70, 0.05, 0.00]]))
            np.save(probs_dir / "probs_t1_swin_tta.npy", np.asarray([[0.10, 0.20, 0.60, 0.10, 0.00]]))

            result = build_submission(
                csv_path=csv_path,
                output_path=output_path,
                probs_dir=probs_dir,
                strategy="weighted-mean",
                model_name="swin_tta",
                raw_weights=["effnet_tta=3", "maxvit_tta=1", "swin_tta=1"],
                manual_labels_path=None,
                manual_source_name=None,
            )

            with output_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows, [{"row_id": "row_1", "class_id": "1"}])
        self.assertEqual(result.model_name, "weighted_mean")


if __name__ == "__main__":
    unittest.main()
