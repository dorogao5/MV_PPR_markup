from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from piglabeler.bot import PigLabelerBot, SessionState
from piglabeler.dataset import DatasetCatalog, PigTask
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


class _FakeBot:
    def __init__(self) -> None:
        self.deleted_messages: list[tuple[int, int]] = []
        self.sent_photos: list[dict[str, object]] = []
        self.sent_messages: list[dict[str, object]] = []
        self._next_message_id = 1

    async def delete_message(self, *, chat_id: int, message_id: int) -> None:
        self.deleted_messages.append((chat_id, message_id))

    async def send_photo(self, **kwargs):
        self.sent_photos.append(kwargs)
        message = type("Message", (), {})()
        message.message_id = self._next_message_id
        self._next_message_id += 1
        return message

    async def send_message(self, **kwargs) -> None:
        self.sent_messages.append(kwargs)


class _FakeContext:
    def __init__(self) -> None:
        self.bot = _FakeBot()


class BotTaskMessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_task_messages_removes_both_task_photos(self) -> None:
        bot = PigLabelerBot.__new__(PigLabelerBot)
        session = SessionState(current_message_ids=[101, 202])
        context = _FakeContext()

        await bot._delete_task_messages(chat_id=777, session=session, context=context)

        self.assertEqual(context.bot.deleted_messages, [(777, 101), (777, 202)])
        self.assertEqual(session.current_message_ids, [])

    async def test_send_task_for_user_sends_original_and_markup_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "frame.jpg"
            rendered_path = root / "rendered.jpg"
            image_path.write_bytes(b"frame")
            rendered_path.write_bytes(b"rendered")

            task = PigTask(
                source_name="test",
                row_id="row_a",
                image_id="frame.jpg",
                image_path=image_path,
                width=1280,
                height=720,
                bbox=(1.0, 2.0, 3.0, 4.0),
                raw_row={"row_id": "row_a", "image_id": "frame.jpg"},
                class_id=None,
                pen=None,
                camera_type=None,
                camera_num=None,
                capture_date=None,
                capture_time=None,
            )

            @dataclass
            class _FakeCatalog:
                task: PigTask
                annotatable_row_ids: list[str]

                def get_image_tasks(self, source_name: str, image_id: str):
                    return [self.task]

                def iter_annotatable_tasks(self):
                    return [self.task]

                def get_task(self, row_id: str):
                    return self.task

                def source_totals(self):
                    return {"test": 1}

                def prediction_for(self, row_id: str):
                    return None

            class _FakeStore:
                def __init__(self) -> None:
                    self.current_annotations = {}

                def is_annotated(self, row_id: str) -> bool:
                    return False

                def count_by_source(self):
                    return {}

            class _FakeRenderer:
                def render_task(self, task, siblings):
                    return rendered_path

            bot = PigLabelerBot.__new__(PigLabelerBot)
            bot.catalog = _FakeCatalog(task, ["row_a"])
            bot.store = _FakeStore()
            bot.renderer = _FakeRenderer()
            bot.help_builder = None
            bot.sessions = {}

            context = _FakeContext()
            await bot._send_task_for_user(
                chat_id=555,
                user_id=42,
                user_name="tester",
                context=context,
            )

        self.assertEqual(len(context.bot.sent_photos), 2)
        self.assertEqual(context.bot.sent_photos[0]["chat_id"], 555)
        self.assertEqual(context.bot.sent_photos[1]["chat_id"], 555)
        session = bot.sessions[42]
        self.assertEqual(session.current_row_id, "row_a")
        self.assertEqual(session.current_message_ids, [1, 2])


if __name__ == "__main__":
    unittest.main()
