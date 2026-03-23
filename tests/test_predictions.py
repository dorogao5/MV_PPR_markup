from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from piglabeler.dataset import DatasetCatalog
from piglabeler.predictions import PredictionQueue


class PredictionQueueTests(unittest.TestCase):
    def test_load_sorts_rows_by_priority_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prediction_queue.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "priority_rank",
                        "row_id",
                        "source_name",
                        "model_name",
                        "predicted_class_id",
                        "confidence",
                        "uncertainty",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "priority_rank": 2,
                            "row_id": "row_b",
                            "source_name": "test",
                            "model_name": "swin_tta",
                            "predicted_class_id": 1,
                            "confidence": 0.8,
                            "uncertainty": 0.2,
                        },
                        {
                            "priority_rank": 1,
                            "row_id": "row_a",
                            "source_name": "test",
                            "model_name": "swin_tta",
                            "predicted_class_id": 3,
                            "confidence": 0.6,
                            "uncertainty": 0.4,
                        },
                    ]
                )

            queue = PredictionQueue.load(path)

        self.assertEqual(queue.ordered_row_ids, ("row_a", "row_b"))
        self.assertEqual(queue.predictions_by_row_id["row_a"].predicted_class_id, 3)

    def test_dataset_catalog_applies_prediction_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "test_images"
            image_dir.mkdir()
            for image_name in ["frame_a.jpg", "frame_b.jpg", "frame_c.jpg"]:
                (image_dir / image_name).write_bytes(b"fake")

            csv_path = root / "test.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["row_id", "image_id", "width", "height", "bbox"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "row_id": "row_a",
                            "image_id": "frame_a.jpg",
                            "width": 10,
                            "height": 10,
                            "bbox": "[0, 0, 1, 1]",
                        },
                        {
                            "row_id": "row_b",
                            "image_id": "frame_b.jpg",
                            "width": 10,
                            "height": 10,
                            "bbox": "[0, 0, 1, 1]",
                        },
                        {
                            "row_id": "row_c",
                            "image_id": "frame_c.jpg",
                            "width": 10,
                            "height": 10,
                            "bbox": "[0, 0, 1, 1]",
                        },
                    ]
                )

            queue_path = root / "prediction_queue.csv"
            with queue_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "priority_rank",
                        "row_id",
                        "source_name",
                        "model_name",
                        "predicted_class_id",
                        "confidence",
                        "uncertainty",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "priority_rank": 1,
                            "row_id": "row_c",
                            "source_name": "test",
                            "model_name": "swin_tta",
                            "predicted_class_id": 2,
                            "confidence": 0.51,
                            "uncertainty": 0.49,
                        },
                        {
                            "priority_rank": 2,
                            "row_id": "row_a",
                            "source_name": "test",
                            "model_name": "swin_tta",
                            "predicted_class_id": 3,
                            "confidence": 0.9,
                            "uncertainty": 0.1,
                        },
                    ]
                )

            catalog = DatasetCatalog.discover(
                root,
                annotate_sources=("test",),
                include_labeled_sources=False,
                prediction_queue=PredictionQueue.load(queue_path),
            )

        self.assertEqual(catalog.annotatable_row_ids, ["row_c", "row_a", "row_b"])
        self.assertEqual(catalog.prediction_for("row_c").predicted_class_id, 2)


if __name__ == "__main__":
    unittest.main()
