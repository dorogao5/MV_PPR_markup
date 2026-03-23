from __future__ import annotations

import asyncio
import csv
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from piglabeler.constants import CLASS_NAMES
from piglabeler.dataset import DatasetCatalog, PigTask

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UndoResult:
    ok: bool
    message: str
    row_id: str | None = None


class AnnotationStore:
    def __init__(self, state_dir: Path) -> None:
        self.annotations_dir = state_dir / "annotations"
        self.exports_dir = state_dir / "exports"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.annotations_dir / "annotation_events.jsonl"
        self.current_path = self.annotations_dir / "current_annotations.csv"

        self._lock = asyncio.Lock()
        self.events: list[dict[str, Any]] = []
        self.current_annotations: dict[str, dict[str, Any]] = {}
        self._load_existing_events()

    def _load_existing_events(self) -> None:
        if not self.events_path.exists():
            return

        with self.events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                self.events.append(event)
                self._apply_event(event)

    def _apply_event(self, event: dict[str, Any]) -> None:
        row_id = event["row_id"]
        if event["kind"] == "set":
            self.current_annotations[row_id] = event
        elif event["kind"] == "delete":
            self.current_annotations.pop(row_id, None)
        else:
            raise RuntimeError(f"Unsupported event kind: {event['kind']}")

    def is_annotated(self, row_id: str) -> bool:
        return row_id in self.current_annotations

    def annotation_for(self, row_id: str) -> dict[str, Any] | None:
        return self.current_annotations.get(row_id)

    def count_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.current_annotations.values():
            source_name = event["source_name"]
            counts[source_name] = counts.get(source_name, 0) + 1
        return counts

    def count_by_class(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for event in self.current_annotations.values():
            class_id = int(event["class_id"])
            counts[class_id] = counts.get(class_id, 0) + 1
        return counts

    async def annotate(
        self,
        task: PigTask,
        class_id: int,
        *,
        annotator_id: int,
        annotator_name: str,
        catalog: DatasetCatalog,
    ) -> dict[str, Any]:
        async with self._lock:
            previous = self.current_annotations.get(task.row_id)
            event = {
                "event_id": uuid.uuid4().hex,
                "kind": "set",
                "timestamp": datetime.now(UTC).isoformat(),
                "row_id": task.row_id,
                "source_name": task.source_name,
                "image_id": task.image_id,
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[int(class_id)],
                "annotator_id": int(annotator_id),
                "annotator_name": annotator_name,
                "previous_class_id": previous["class_id"] if previous else None,
            }
            self._append_event(event)
            self._apply_event(event)
            self._write_snapshots(catalog)
            return event

    async def undo_last_for_user(
        self,
        *,
        annotator_id: int,
        annotator_name: str,
        catalog: DatasetCatalog,
    ) -> UndoResult:
        async with self._lock:
            located = self._find_last_effective_set_event(annotator_id)
            if located is None:
                return UndoResult(False, "У тебя нет последней подтвержденной разметки для отката.")

            index, event = located
            row_id = event["row_id"]
            previous = self._find_effective_state_before(row_id, index)

            if previous is None:
                undo_event = {
                    "event_id": uuid.uuid4().hex,
                    "kind": "delete",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "row_id": row_id,
                    "source_name": event["source_name"],
                    "image_id": event["image_id"],
                    "annotator_id": int(annotator_id),
                    "annotator_name": annotator_name,
                    "undo_of": event["event_id"],
                }
            else:
                undo_event = {
                    "event_id": uuid.uuid4().hex,
                    "kind": "set",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "row_id": row_id,
                    "source_name": previous["source_name"],
                    "image_id": previous["image_id"],
                    "class_id": int(previous["class_id"]),
                    "class_name": previous["class_name"],
                    "annotator_id": int(annotator_id),
                    "annotator_name": annotator_name,
                    "undo_of": event["event_id"],
                    "restored_from_event_id": previous["event_id"],
                }

            self._append_event(undo_event)
            self._apply_event(undo_event)
            self._write_snapshots(catalog)
            return UndoResult(True, "Последняя разметка откатена.", row_id=row_id)

    def _find_last_effective_set_event(
        self,
        annotator_id: int,
    ) -> tuple[int, dict[str, Any]] | None:
        for index in range(len(self.events) - 1, -1, -1):
            event = self.events[index]
            if event["kind"] != "set":
                continue
            if int(event["annotator_id"]) != int(annotator_id):
                continue

            row_id = event["row_id"]
            last_row_event = self._last_event_for_row(row_id)
            if last_row_event and last_row_event["event_id"] == event["event_id"]:
                return index, event
        return None

    def _last_event_for_row(self, row_id: str) -> dict[str, Any] | None:
        for event in reversed(self.events):
            if event["row_id"] == row_id:
                return event
        return None

    def _find_effective_state_before(
        self,
        row_id: str,
        index: int,
    ) -> dict[str, Any] | None:
        for candidate_index in range(index - 1, -1, -1):
            event = self.events[candidate_index]
            if event["row_id"] != row_id:
                continue
            if event["kind"] == "set":
                return event
            if event["kind"] == "delete":
                return None
        return None

    def _append_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.events.append(event)

    def _write_snapshots(self, catalog: DatasetCatalog) -> None:
        current_rows: list[dict[str, Any]] = []
        for row_id, event in sorted(
            self.current_annotations.items(),
            key=lambda item: (item[1]["source_name"], item[1]["image_id"], item[0]),
        ):
            task = catalog.get_task(row_id)
            current_rows.append(
                {
                    "row_id": row_id,
                    "source_name": event["source_name"],
                    "image_id": event["image_id"],
                    "class_id": event["class_id"],
                    "class_name": event["class_name"],
                    "annotated_at": event["timestamp"],
                    "annotator_id": event["annotator_id"],
                    "annotator_name": event["annotator_name"],
                    "width": task.width,
                    "height": task.height,
                    "bbox": list(task.bbox),
                }
            )

        self._write_csv_atomic(
            self.current_path,
            current_rows,
            [
                "row_id",
                "source_name",
                "image_id",
                "class_id",
                "class_name",
                "annotated_at",
                "annotator_id",
                "annotator_name",
                "width",
                "height",
                "bbox",
            ],
        )

        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in current_rows:
            by_source.setdefault(row["source_name"], []).append(row)

        for source_name in catalog.source_totals():
            source_tasks = [
                catalog.get_task(row["row_id"])
                for row in by_source.get(source_name, [])
            ]
            export_rows = []
            submission_rows = []
            for task in source_tasks:
                event = self.current_annotations[task.row_id]
                merged_row = dict(task.raw_row)
                merged_row["class_id"] = event["class_id"]
                merged_row["class_name"] = event["class_name"]
                merged_row["annotated_at"] = event["timestamp"]
                merged_row["annotator_id"] = event["annotator_id"]
                merged_row["annotator_name"] = event["annotator_name"]
                export_rows.append(merged_row)
                submission_rows.append(
                    {
                        "row_id": task.row_id,
                        "class_id": event["class_id"],
                    }
                )

            export_fields = list(source_tasks[0].raw_row.keys()) if source_tasks else [
                "row_id",
                "image_id",
                "width",
                "height",
                "bbox",
            ]
            if "class_id" not in export_fields:
                export_fields.append("class_id")
            export_fields.extend(
                field
                for field in ["class_name", "annotated_at", "annotator_id", "annotator_name"]
                if field not in export_fields
            )

            self._write_csv_atomic(
                self.exports_dir / f"{source_name}_manual_labels.csv",
                export_rows,
                export_fields,
            )
            self._write_csv_atomic(
                self.exports_dir / f"{source_name}_submission.csv",
                submission_rows,
                ["row_id", "class_id"],
            )

    def _write_csv_atomic(
        self,
        path: Path,
        rows: list[dict[str, Any]],
        fieldnames: list[str],
    ) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
