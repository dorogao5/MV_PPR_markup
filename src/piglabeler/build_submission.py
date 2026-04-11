from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from piglabeler.prepare_queue import DEFAULT_PROB_PATHS


@dataclass(frozen=True)
class SubmissionBuildResult:
    total_rows: int
    manual_rows_loaded: int
    manual_rows_applied: int
    manual_rows_missing_in_base: int
    model_name: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Kaggle submission from precomputed probabilities and "
            "optionally override selected rows with manual annotations."
        )
    )
    parser.add_argument("--csv", type=Path, default=Path("test.csv"), help="Source CSV with row_id order.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/submission_blend_manual.csv"),
        help="Where to write the final submission CSV.",
    )
    parser.add_argument(
        "--probs-dir",
        type=Path,
        default=Path("probs"),
        help="Directory with probability .npy files.",
    )
    parser.add_argument(
        "--strategy",
        choices=("best-model", "ensemble-mean", "weighted-mean"),
        default="ensemble-mean",
        help="How to combine the probability files.",
    )
    parser.add_argument(
        "--model",
        default="swin_tta",
        help="Model to use when --strategy=best-model.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="MODEL=WEIGHT",
        help="Model weight for --strategy=weighted-mean. Can be passed multiple times.",
    )
    parser.add_argument(
        "--manual-labels",
        type=Path,
        default=None,
        help=(
            "Optional CSV with manual labels. Supports current_annotations.csv, "
            "*_manual_labels.csv, or *_submission.csv."
        ),
    )
    parser.add_argument(
        "--manual-source",
        default=None,
        help=(
            "Optional source_name filter for manual labels. "
            "Defaults to the source CSV stem when the manual-label file contains source_name."
        ),
    )
    return parser


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    if "row_id" not in rows[0]:
        raise RuntimeError(f"{csv_path} must contain row_id column.")
    return rows


def _normalize_discovered_model_name(path: Path) -> str:
    stem = path.stem
    for prefix in ("probs_t1_", "probs_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _resolve_probability_paths(probs_dir: Path) -> OrderedDict[str, Path]:
    default_paths = OrderedDict(
        (model_name, probs_dir / path.name) for model_name, path in DEFAULT_PROB_PATHS.items()
    )
    if all(path.exists() for path in default_paths.values()):
        return default_paths

    discovered_paths = sorted(probs_dir.glob("*.npy"))
    if not discovered_paths:
        raise FileNotFoundError(f"No probability files found in {probs_dir}")

    resolved: OrderedDict[str, Path] = OrderedDict()
    for path in discovered_paths:
        model_name = _normalize_discovered_model_name(path)
        if model_name in resolved:
            raise RuntimeError(
                f"Duplicate model name {model_name!r} after discovery in {probs_dir}. "
                "Rename the .npy files or use the default filenames."
            )
        resolved[model_name] = path
    return resolved


def _load_probabilities(prob_paths: OrderedDict[str, Path]):
    import numpy as np

    arrays = OrderedDict()
    expected_shape = None
    for model_name, path in prob_paths.items():
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


def _parse_weight_overrides(raw_weights: list[str], model_names: list[str]) -> dict[str, float]:
    weights = {model_name: 1.0 for model_name in model_names}
    for item in raw_weights:
        if "=" not in item:
            raise RuntimeError(f"Invalid --weight value {item!r}. Expected MODEL=WEIGHT.")
        model_name, raw_weight = item.split("=", 1)
        model_name = model_name.strip()
        if model_name not in weights:
            raise RuntimeError(
                f"Unknown model {model_name!r} in --weight. Available models: {', '.join(model_names)}."
            )
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise RuntimeError(f"Invalid weight {raw_weight!r} for model {model_name!r}.") from exc
        if weight < 0:
            raise RuntimeError(f"Weight must be non-negative for model {model_name!r}.")
        weights[model_name] = weight
    if sum(weights.values()) <= 0:
        raise RuntimeError("At least one model weight must be positive.")
    return weights


def _select_probabilities(
    arrays,
    *,
    strategy: str,
    model_name: str,
    raw_weights: list[str],
):
    import numpy as np

    if strategy == "best-model":
        if model_name not in arrays:
            raise RuntimeError(
                f"Unknown model {model_name!r}. Available models: {', '.join(arrays.keys())}."
            )
        return arrays[model_name], model_name

    ordered_items = list(arrays.items())
    stacked = np.stack([array for _, array in ordered_items], axis=0)

    if strategy == "ensemble-mean":
        combined = stacked.mean(axis=0)
        return combined, "ensemble_mean"

    weights = _parse_weight_overrides(raw_weights, [name for name, _ in ordered_items])
    weight_vector = np.asarray([weights[name] for name, _ in ordered_items], dtype=float)
    combined = np.tensordot(weight_vector, stacked, axes=(0, 0)) / weight_vector.sum()
    return combined, "weighted_mean"


def _load_manual_labels(
    manual_labels_path: Path | None,
    *,
    default_source_name: str,
    manual_source_name: str | None,
) -> dict[str, int]:
    if manual_labels_path is None:
        return {}

    with manual_labels_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "row_id" not in reader.fieldnames or "class_id" not in reader.fieldnames:
            raise RuntimeError(f"{manual_labels_path} must contain row_id and class_id columns.")

        source_filter = manual_source_name
        if source_filter is None and "source_name" in reader.fieldnames:
            source_filter = default_source_name

        overrides: dict[str, int] = {}
        for row in reader:
            if not row:
                continue
            row_id = (row.get("row_id") or "").strip()
            class_id_raw = (row.get("class_id") or "").strip()
            if not row_id or not class_id_raw:
                continue
            if source_filter is not None and "source_name" in row:
                if (row.get("source_name") or "").strip() != source_filter:
                    continue
            overrides[row_id] = int(class_id_raw)
        return overrides


def build_submission(
    *,
    csv_path: Path,
    output_path: Path,
    probs_dir: Path,
    strategy: str,
    model_name: str,
    raw_weights: list[str],
    manual_labels_path: Path | None,
    manual_source_name: str | None,
) -> SubmissionBuildResult:
    import numpy as np

    rows = _load_rows(csv_path)
    prob_paths = _resolve_probability_paths(probs_dir)
    arrays = _load_probabilities(prob_paths)
    probabilities, selected_model_name = _select_probabilities(
        arrays,
        strategy=strategy,
        model_name=model_name,
        raw_weights=raw_weights,
    )

    if len(rows) != probabilities.shape[0]:
        raise RuntimeError(
            f"Row count mismatch: {csv_path} contains {len(rows)} rows, "
            f"but the selected probabilities contain {probabilities.shape[0]} rows."
        )

    submission_rows = []
    index_by_row_id = {}
    for index, row in enumerate(rows):
        predicted_class_id = int(np.asarray(probabilities[index]).argmax())
        submission_rows.append(
            {
                "row_id": row["row_id"],
                "class_id": predicted_class_id,
            }
        )
        index_by_row_id[row["row_id"]] = index

    manual_labels = _load_manual_labels(
        manual_labels_path,
        default_source_name=csv_path.stem,
        manual_source_name=manual_source_name,
    )

    manual_rows_applied = 0
    manual_rows_missing_in_base = 0
    for row_id, class_id in manual_labels.items():
        index = index_by_row_id.get(row_id)
        if index is None:
            manual_rows_missing_in_base += 1
            continue
        if submission_rows[index]["class_id"] != class_id:
            manual_rows_applied += 1
        submission_rows[index]["class_id"] = class_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "class_id"])
        writer.writeheader()
        writer.writerows(submission_rows)

    return SubmissionBuildResult(
        total_rows=len(submission_rows),
        manual_rows_loaded=len(manual_labels),
        manual_rows_applied=manual_rows_applied,
        manual_rows_missing_in_base=manual_rows_missing_in_base,
        model_name=selected_model_name,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = build_submission(
        csv_path=args.csv,
        output_path=args.output,
        probs_dir=args.probs_dir,
        strategy=args.strategy,
        model_name=args.model,
        raw_weights=args.weight,
        manual_labels_path=args.manual_labels,
        manual_source_name=args.manual_source,
    )

    print(
        f"Wrote {result.total_rows} rows to {args.output} "
        f"using {result.model_name}. "
        f"Loaded {result.manual_rows_loaded} manual labels, "
        f"applied {result.manual_rows_applied}, "
        f"skipped {result.manual_rows_missing_in_base} unknown row_id values."
    )


if __name__ == "__main__":
    main()
