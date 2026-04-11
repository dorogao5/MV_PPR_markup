from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from piglabeler.constants import CLASS_NAMES
from piglabeler.prepare_queue import DEFAULT_PROB_PATHS, _load_rows

AMBIGUOUS_TOP2_PAIRS = {(0, 1), (3, 4)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a full prediction queue for the Telegram bot, prioritizing "
            "rows with disagreement between strong historical submissions and "
            "the current probability models."
        )
    )
    parser.add_argument("--csv", type=Path, default=Path("test.csv"), help="Source CSV with row_id order.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/prediction_queue.csv"),
        help="Where to write the prepared queue CSV.",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Override source_name stored in the queue. Defaults to the CSV stem.",
    )
    parser.add_argument(
        "--old-submission",
        action="append",
        default=[],
        metavar="PATH",
        help="Historical strong submission CSV to use as an additional disagreement signal.",
    )
    return parser


def _discover_old_submissions(explicit_paths: list[str]) -> list[Path]:
    if explicit_paths:
        return [Path(item).expanduser().resolve() for item in explicit_paths]

    discovered = sorted(Path("submissions/old").glob("*.csv"))
    return [path.resolve() for path in discovered]


def _load_probabilities():
    import numpy as np

    arrays = OrderedDict()
    expected_shape = None
    for model_name, path in DEFAULT_PROB_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Probability file not found: {path}")
        array = np.load(path)
        if array.ndim != 2:
            raise RuntimeError(f"{path} must be a 2D array, got shape {array.shape}")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise RuntimeError(
                f"All probability arrays must have the same shape. "
                f"Expected {expected_shape}, got {array.shape} for {path}"
            )
        arrays[model_name] = array
    return arrays


def _load_submission_predictions(paths: list[Path]) -> OrderedDict[str, dict[str, int]]:
    loaded: OrderedDict[str, dict[str, int]] = OrderedDict()
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "row_id" not in reader.fieldnames or "class_id" not in reader.fieldnames:
                raise RuntimeError(f"{path} must contain row_id and class_id columns.")
            predictions = {row["row_id"]: int(row["class_id"]) for row in reader}
        loaded[path.stem] = predictions
    return loaded


def prepare_disagreement_queue(
    *,
    csv_path: Path,
    output_path: Path,
    source_name: str | None,
    old_submission_paths: list[Path],
) -> int:
    import numpy as np

    rows = _load_rows(csv_path)
    arrays = _load_probabilities()
    mean_probabilities = sum(arrays.values()) / len(arrays)
    old_submission_predictions = _load_submission_predictions(old_submission_paths)

    if len(rows) != mean_probabilities.shape[0]:
        raise RuntimeError(
            f"Row count mismatch: {csv_path} contains {len(rows)} rows, "
            f"but the selected probabilities contain {mean_probabilities.shape[0]} rows."
        )

    expected_row_ids = {row["row_id"] for row in rows}
    for submission_name, predictions in old_submission_predictions.items():
        missing = expected_row_ids.difference(predictions)
        if missing:
            sample = ", ".join(sorted(missing)[:3])
            raise RuntimeError(
                f"Historical submission {submission_name} is missing {len(missing)} row_id values. "
                f"Sample: {sample}"
            )

    source_name = source_name or csv_path.stem
    class_count = mean_probabilities.shape[1]
    prob_columns = [f"prob_{class_id}" for class_id in range(class_count)]
    old_submission_names = list(old_submission_predictions)

    ranked_rows = []
    for index, row in enumerate(rows):
        row_probs = mean_probabilities[index]
        mean_order = np.argsort(row_probs)[::-1]
        predicted_class_id = int(mean_order[0])
        confidence = float(row_probs[predicted_class_id])
        second_best_class_id = int(mean_order[1]) if len(mean_order) > 1 else predicted_class_id
        margin = (
            float(row_probs[mean_order[0]] - row_probs[mean_order[1]])
            if len(mean_order) > 1
            else float(row_probs[mean_order[0]])
        )

        model_predictions = {
            model_name: int(probabilities[index].argmax())
            for model_name, probabilities in arrays.items()
        }
        model_unique_pred_count = len(set(model_predictions.values()))

        old_predictions = {
            submission_name: predictions[row["row_id"]]
            for submission_name, predictions in old_submission_predictions.items()
        }
        old_unique_pred_count = len(set(old_predictions.values())) if old_predictions else 0
        any_old_vs_mean = int(any(pred != predicted_class_id for pred in old_predictions.values()))
        ambiguous_top2 = int(tuple(sorted((predicted_class_id, second_best_class_id))) in AMBIGUOUS_TOP2_PAIRS)

        ranked_rows.append(
            {
                "row_id": row["row_id"],
                "source_name": source_name,
                "image_id": row["image_id"],
                "model_name": (
                    f"disagreement_priority_mean_{len(old_submission_names)}_old_submissions"
                    if old_submission_names
                    else "disagreement_priority_mean"
                ),
                "predicted_class_id": predicted_class_id,
                "predicted_class_name": CLASS_NAMES[predicted_class_id],
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "margin": margin,
                "second_best_class_id": second_best_class_id,
                "second_best_class_name": CLASS_NAMES[second_best_class_id],
                "ambiguous_top2": ambiguous_top2,
                "model_unique_pred_count": model_unique_pred_count,
                "old_unique_pred_count": old_unique_pred_count,
                "any_old_vs_mean": any_old_vs_mean,
                **{f"{model_name}_class_id": class_id for model_name, class_id in model_predictions.items()},
                **{f"{submission_name}_class_id": class_id for submission_name, class_id in old_predictions.items()},
                **{prob_columns[class_id]: float(row_probs[class_id]) for class_id in range(class_count)},
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            -row["old_unique_pred_count"],
            -row["model_unique_pred_count"],
            -row["ambiguous_top2"],
            -row["any_old_vs_mean"],
            row["margin"],
            row["confidence"],
            row["row_id"],
        )
    )

    for priority_rank, row in enumerate(ranked_rows, start=1):
        row["priority_rank"] = priority_rank

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority_rank",
        "row_id",
        "source_name",
        "image_id",
        "model_name",
        "predicted_class_id",
        "predicted_class_name",
        "confidence",
        "uncertainty",
        "margin",
        "second_best_class_id",
        "second_best_class_name",
        "ambiguous_top2",
        "model_unique_pred_count",
        "old_unique_pred_count",
        "any_old_vs_mean",
        *[f"{model_name}_class_id" for model_name in arrays],
        *[f"{submission_name}_class_id" for submission_name in old_submission_names],
        *prob_columns,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)

    return len(ranked_rows)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    old_submission_paths = _discover_old_submissions(args.old_submission)
    total_rows = prepare_disagreement_queue(
        csv_path=args.csv,
        output_path=args.output,
        source_name=args.source_name,
        old_submission_paths=old_submission_paths,
    )
    print(
        f"Wrote {total_rows} queue rows to {args.output} "
        f"using {len(old_submission_paths)} historical submissions."
    )


if __name__ == "__main__":
    main()
