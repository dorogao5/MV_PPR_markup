from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionHint:
    row_id: str
    source_name: str
    model_name: str
    predicted_class_id: int
    confidence: float
    uncertainty: float
    priority_rank: int


@dataclass(frozen=True)
class PredictionQueue:
    ordered_row_ids: tuple[str, ...]
    predictions_by_row_id: dict[str, PredictionHint]

    @classmethod
    def load(cls, path: Path) -> "PredictionQueue":
        required_columns = {
            "row_id",
            "source_name",
            "model_name",
            "predicted_class_id",
            "confidence",
            "uncertainty",
            "priority_rank",
        }

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = required_columns - fieldnames
            if missing_columns:
                raise RuntimeError(
                    f"Prediction queue {path} is missing required columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )

            predictions: dict[str, PredictionHint] = {}
            ordered_predictions: list[PredictionHint] = []
            for row in reader:
                row_id = (row.get("row_id") or "").strip()
                if not row_id:
                    raise RuntimeError(f"Prediction queue {path} contains an empty row_id.")
                if row_id in predictions:
                    raise RuntimeError(f"Prediction queue {path} contains duplicate row_id {row_id}.")

                hint = PredictionHint(
                    row_id=row_id,
                    source_name=(row.get("source_name") or "").strip(),
                    model_name=(row.get("model_name") or "").strip(),
                    predicted_class_id=int(row["predicted_class_id"]),
                    confidence=float(row["confidence"]),
                    uncertainty=float(row["uncertainty"]),
                    priority_rank=int(row["priority_rank"]),
                )
                predictions[row_id] = hint
                ordered_predictions.append(hint)

        ordered_predictions.sort(key=lambda hint: (hint.priority_rank, hint.row_id))
        return cls(
            ordered_row_ids=tuple(hint.row_id for hint in ordered_predictions),
            predictions_by_row_id=predictions,
        )


def resolve_prediction_queue_path(
    *,
    explicit_path: Path | None,
    data_dir: Path,
) -> Path | None:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prediction queue file does not exist: {path}")
        return path

    candidates = [
        data_dir / "prediction_queue.csv",
        data_dir / "test_prediction_queue.csv",
        Path("predictions/prediction_queue.csv").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None

