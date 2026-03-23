from __future__ import annotations

import argparse
import csv
from pathlib import Path

from piglabeler.constants import CLASS_NAMES

DEFAULT_PROB_PATHS = {
    "effnet_tta": Path("probs/probs_t1_effnet_tta.npy"),
    "maxvit_tta": Path("probs/probs_t1_maxvit_tta.npy"),
    "swin_tta": Path("probs/probs_t1_swin_tta.npy"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a pre-sorted prediction queue for the Telegram bot. "
            "The server only reads the generated CSV and does not load model outputs."
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
        "--strategy",
        choices=("best-model", "ensemble-mean"),
        default="ensemble-mean",
        help="How to combine the precomputed probability files.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(DEFAULT_PROB_PATHS),
        default="swin_tta",
        help="Model to use when --strategy=best-model.",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Override source_name stored in the queue. Defaults to the CSV stem.",
    )
    return parser


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    if "row_id" not in rows[0] or "image_id" not in rows[0]:
        raise RuntimeError(f"{csv_path} must contain row_id and image_id columns.")
    return rows


def _load_probabilities():
    import numpy as np

    arrays = {}
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


def _select_probabilities(arrays, strategy: str, model_name: str):
    if strategy == "best-model":
        return arrays[model_name], model_name
    combined = sum(arrays.values()) / len(arrays)
    return combined, "ensemble_mean_effnet_maxvit_swin"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    arrays = _load_probabilities()
    probabilities, queue_model_name = _select_probabilities(arrays, args.strategy, args.model)

    if len(rows) != probabilities.shape[0]:
        raise RuntimeError(
            f"Row count mismatch: {args.csv} contains {len(rows)} rows, "
            f"but the selected probabilities contain {probabilities.shape[0]} rows."
        )

    source_name = args.source_name or args.csv.stem
    class_count = probabilities.shape[1]
    prob_columns = [f"prob_{class_id}" for class_id in range(class_count)]

    ranked_rows = []
    for index, row in enumerate(rows):
        row_probs = probabilities[index]
        predicted_class_id = int(row_probs.argmax())
        confidence = float(row_probs[predicted_class_id])
        sorted_probs = sorted((float(value) for value in row_probs), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        ranked_rows.append(
            {
                "row_id": row["row_id"],
                "source_name": source_name,
                "image_id": row["image_id"],
                "model_name": queue_model_name,
                "predicted_class_id": predicted_class_id,
                "predicted_class_name": CLASS_NAMES[predicted_class_id],
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "margin": margin,
                **{
                    prob_columns[class_id]: float(row_probs[class_id])
                    for class_id in range(class_count)
                },
            }
        )

    ranked_rows.sort(key=lambda row: (row["confidence"], row["margin"], row["row_id"]))
    for priority_rank, row in enumerate(ranked_rows, start=1):
        row["priority_rank"] = priority_rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        *prob_columns,
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked_rows)

    print(
        f"Wrote {len(ranked_rows)} queue rows to {args.output} "
        f"using {queue_model_name} ({args.strategy})."
    )


if __name__ == "__main__":
    main()
