from __future__ import annotations

import ast
import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from piglabeler.constants import REQUIRED_DATA_COLUMNS
from piglabeler.predictions import PredictionHint, PredictionQueue

LOGGER = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"^(?P<pen>pen\d+)_(?P<camera_type>\w+)_(?P<camera_num>cam\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.jpg$"
)


@dataclass(frozen=True)
class PigTask:
    source_name: str
    row_id: str
    image_id: str
    image_path: Path
    width: int
    height: int
    bbox: tuple[float, float, float, float]
    raw_row: dict[str, str]
    class_id: int | None
    pen: str | None
    camera_type: str | None
    camera_num: str | None
    capture_date: str | None
    capture_time: str | None

    @property
    def camera_view(self) -> str | None:
        if not self.camera_type or not self.camera_num:
            return None
        return f"{self.camera_type}_{self.camera_num}"

    @property
    def bbox_area(self) -> float:
        return self.bbox[2] * self.bbox[3]


@dataclass(frozen=True)
class SourceSummary:
    name: str
    csv_path: Path
    image_dir: Path
    has_labels: bool
    total_rows: int
    total_images: int


class DatasetCatalog:
    def __init__(
        self,
        *,
        tasks_by_row_id: dict[str, PigTask],
        tasks_by_image: dict[tuple[str, str], list[PigTask]],
        sources: dict[str, SourceSummary],
        annotatable_row_ids: list[str],
        prediction_hints: dict[str, PredictionHint],
    ) -> None:
        self.tasks_by_row_id = tasks_by_row_id
        self.tasks_by_image = tasks_by_image
        self.sources = sources
        self.annotatable_row_ids = annotatable_row_ids
        self.prediction_hints = prediction_hints

    @classmethod
    def discover(
        cls,
        data_dir: Path,
        *,
        annotate_sources: tuple[str, ...] | None,
        include_labeled_sources: bool,
        prediction_queue: PredictionQueue | None = None,
    ) -> "DatasetCatalog":
        tasks_by_row_id: dict[str, PigTask] = {}
        tasks_by_image: dict[tuple[str, str], list[PigTask]] = defaultdict(list)
        sources: dict[str, SourceSummary] = {}
        annotatable_row_ids: list[str] = []

        csv_files = sorted(path for path in data_dir.glob("*.csv") if path.is_file())
        for csv_path in csv_files:
            source_name = csv_path.stem
            if source_name == "sample_submission":
                continue

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                if not all(column in reader.fieldnames for column in REQUIRED_DATA_COLUMNS):
                    continue

                image_dir = data_dir / f"{source_name}_images"
                if not image_dir.exists():
                    LOGGER.warning("Skipping %s: image dir %s does not exist", source_name, image_dir)
                    continue

                has_labels = "class_id" in reader.fieldnames
                wanted = annotate_sources is None or source_name in annotate_sources
                allow_source_for_annotation = wanted and (include_labeled_sources or not has_labels)

                rows_in_source = 0
                images_in_source: set[str] = set()
                for row in reader:
                    bbox = tuple(float(value) for value in ast.literal_eval(row["bbox"]))
                    image_id = row["image_id"]
                    image_path = image_dir / image_id
                    if not image_path.exists():
                        raise FileNotFoundError(
                            f"Missing image for row {row['row_id']}: {image_path}"
                        )

                    match = FILENAME_RE.match(image_id)
                    metadata = match.groupdict() if match else {}
                    class_id = int(row["class_id"]) if row.get("class_id") not in (None, "") else None
                    task = PigTask(
                        source_name=source_name,
                        row_id=row["row_id"],
                        image_id=image_id,
                        image_path=image_path,
                        width=int(float(row["width"])),
                        height=int(float(row["height"])),
                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        raw_row=dict(row),
                        class_id=class_id,
                        pen=metadata.get("pen"),
                        camera_type=metadata.get("camera_type"),
                        camera_num=metadata.get("camera_num"),
                        capture_date=metadata.get("date"),
                        capture_time=metadata.get("time"),
                    )

                    rows_in_source += 1
                    images_in_source.add(image_id)

                    if task.row_id in tasks_by_row_id:
                        existing = tasks_by_row_id[task.row_id]
                        if existing.class_id is not None and task.class_id is not None:
                            LOGGER.debug(
                                "Skipping duplicate labeled row_id %s from source %s",
                                task.row_id,
                                source_name,
                            )
                            continue
                        raise RuntimeError(f"Duplicate row_id found in annotatable sources: {task.row_id}")

                    tasks_by_row_id[task.row_id] = task
                    tasks_by_image[(source_name, image_id)].append(task)
                    if allow_source_for_annotation:
                        annotatable_row_ids.append(task.row_id)

                sources[source_name] = SourceSummary(
                    name=source_name,
                    csv_path=csv_path,
                    image_dir=image_dir,
                    has_labels=has_labels,
                    total_rows=rows_in_source,
                    total_images=len(images_in_source),
                )

        annotatable_row_ids.sort(
            key=lambda row_id: (
                tasks_by_row_id[row_id].source_name,
                tasks_by_row_id[row_id].image_id,
                tasks_by_row_id[row_id].row_id,
            )
        )
        prediction_hints = cls._collect_prediction_hints(
            tasks_by_row_id=tasks_by_row_id,
            prediction_queue=prediction_queue,
        )
        annotatable_row_ids = cls._apply_prediction_order(
            annotatable_row_ids=annotatable_row_ids,
            tasks_by_row_id=tasks_by_row_id,
            prediction_queue=prediction_queue,
        )
        if not annotatable_row_ids:
            LOGGER.warning("No annotatable sources were discovered in %s", data_dir)

        return cls(
            tasks_by_row_id=tasks_by_row_id,
            tasks_by_image=dict(tasks_by_image),
            sources=sources,
            annotatable_row_ids=annotatable_row_ids,
            prediction_hints=prediction_hints,
        )

    def get_task(self, row_id: str) -> PigTask:
        return self.tasks_by_row_id[row_id]

    def get_image_tasks(self, source_name: str, image_id: str) -> list[PigTask]:
        return list(self.tasks_by_image[(source_name, image_id)])

    def prediction_for(self, row_id: str) -> PredictionHint | None:
        return self.prediction_hints.get(row_id)

    def iter_annotatable_tasks(self) -> list[PigTask]:
        return [self.tasks_by_row_id[row_id] for row_id in self.annotatable_row_ids]

    def source_totals(self) -> dict[str, int]:
        totals: dict[str, int] = defaultdict(int)
        for row_id in self.annotatable_row_ids:
            totals[self.tasks_by_row_id[row_id].source_name] += 1
        return dict(totals)

    def source_rows(self, source_name: str) -> list[PigTask]:
        return [task for task in self.tasks_by_row_id.values() if task.source_name == source_name]

    def labeled_reference_tasks(self) -> list[PigTask]:
        return [
            task
            for task in self.tasks_by_row_id.values()
            if task.class_id is not None and self.sources[task.source_name].has_labels
        ]

    @staticmethod
    def _collect_prediction_hints(
        *,
        tasks_by_row_id: dict[str, PigTask],
        prediction_queue: PredictionQueue | None,
    ) -> dict[str, PredictionHint]:
        if prediction_queue is None:
            return {}
        return {
            row_id: hint
            for row_id, hint in prediction_queue.predictions_by_row_id.items()
            if row_id in tasks_by_row_id
        }

    @staticmethod
    def _apply_prediction_order(
        *,
        annotatable_row_ids: list[str],
        tasks_by_row_id: dict[str, PigTask],
        prediction_queue: PredictionQueue | None,
    ) -> list[str]:
        if prediction_queue is None:
            return annotatable_row_ids

        annotatable_set = set(annotatable_row_ids)
        prioritized: list[str] = []
        seen: set[str] = set()
        missing_row_ids = 0
        non_annotatable_row_ids = 0

        for row_id in prediction_queue.ordered_row_ids:
            if row_id not in tasks_by_row_id:
                missing_row_ids += 1
                continue
            if row_id not in annotatable_set:
                non_annotatable_row_ids += 1
                continue
            prioritized.append(row_id)
            seen.add(row_id)

        if missing_row_ids:
            LOGGER.warning(
                "Prediction queue contains %s row_ids that are not present in the dataset.",
                missing_row_ids,
            )
        if non_annotatable_row_ids:
            LOGGER.info(
                "Prediction queue contains %s row_ids outside annotatable sources.",
                non_annotatable_row_ids,
            )

        if not prioritized:
            LOGGER.warning("Prediction queue was loaded, but no annotatable rows matched it.")
            return annotatable_row_ids

        remainder = [row_id for row_id in annotatable_row_ids if row_id not in seen]
        return prioritized + remainder
