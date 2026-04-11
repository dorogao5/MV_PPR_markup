from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from piglabeler.prepare_disagreement_queue import prepare_disagreement_queue


class PrepareDisagreementQueueTests(unittest.TestCase):
    def test_prepare_disagreement_queue_prioritizes_old_submission_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "test.csv"
            output_path = root / "prediction_queue.csv"
            old_a_path = root / "old_a.csv"
            old_b_path = root / "old_b.csv"

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

            for path, classes in [
                (old_a_path, [4, 1, 3]),
                (old_b_path, [0, 1, 3]),
            ]:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["row_id", "class_id"])
                    writer.writeheader()
                    writer.writerows(
                        [
                            {"row_id": "row_1", "class_id": classes[0]},
                            {"row_id": "row_2", "class_id": classes[1]},
                            {"row_id": "row_3", "class_id": classes[2]},
                        ]
                    )

            arrays = {
                "effnet_tta": np.asarray(
                    [
                        [0.05, 0.05, 0.10, 0.40, 0.40],
                        [0.05, 0.45, 0.05, 0.40, 0.05],
                        [0.05, 0.10, 0.05, 0.75, 0.05],
                    ]
                ),
                "maxvit_tta": np.asarray(
                    [
                        [0.05, 0.05, 0.40, 0.05, 0.45],
                        [0.05, 0.40, 0.05, 0.45, 0.05],
                        [0.05, 0.10, 0.05, 0.70, 0.10],
                    ]
                ),
                "swin_tta": np.asarray(
                    [
                        [0.40, 0.05, 0.05, 0.10, 0.40],
                        [0.05, 0.43, 0.05, 0.42, 0.05],
                        [0.05, 0.10, 0.05, 0.72, 0.08],
                    ]
                ),
            }

            with patch(
                "piglabeler.prepare_disagreement_queue._load_probabilities",
                return_value=arrays,
            ):
                total_rows = prepare_disagreement_queue(
                    csv_path=csv_path,
                    output_path=output_path,
                    source_name=None,
                    old_submission_paths=[old_a_path, old_b_path],
                )

            with output_path.open("r", newline="", encoding="utf-8") as handle:
                queue_rows = list(csv.DictReader(handle))

        self.assertEqual(total_rows, 3)
        self.assertEqual([row["row_id"] for row in queue_rows], ["row_1", "row_2", "row_3"])
        self.assertEqual(queue_rows[0]["old_unique_pred_count"], "2")
        self.assertEqual(queue_rows[0]["model_unique_pred_count"], "3")
        self.assertEqual(queue_rows[0]["priority_rank"], "1")


if __name__ == "__main__":
    unittest.main()
