from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatchSubmissionResult:
    total_rows: int
    manual_rows_loaded: int
    manual_rows_applied: int
    manual_rows_missing_in_base: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Patch an existing Kaggle submission with manual labels by row_id."
        )
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        required=True,
        help="Existing submission CSV with row_id,class_id.",
    )
    parser.add_argument(
        "--manual-labels",
        type=Path,
        required=True,
        help=(
            "CSV with manual labels. Supports current_annotations.csv, "
            "*_manual_labels.csv, or *_submission.csv."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the patched submission CSV.",
    )
    parser.add_argument(
        "--manual-source",
        default=None,
        help=(
            "Optional source_name filter for manual labels. "
            "Defaults to the base submission stem when the manual-label file contains source_name."
        ),
    )
    return parser


def _load_submission_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "row_id" not in reader.fieldnames or "class_id" not in reader.fieldnames:
            raise RuntimeError(f"{path} must contain row_id and class_id columns.")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def _load_manual_labels(
    manual_labels_path: Path,
    *,
    default_source_name: str,
    manual_source_name: str | None,
) -> dict[str, int]:
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


def patch_submission(
    *,
    base_submission_path: Path,
    manual_labels_path: Path,
    output_path: Path,
    manual_source_name: str | None,
) -> PatchSubmissionResult:
    submission_rows = _load_submission_rows(base_submission_path)
    manual_labels = _load_manual_labels(
        manual_labels_path,
        default_source_name=base_submission_path.stem,
        manual_source_name=manual_source_name,
    )

    index_by_row_id = {row["row_id"]: index for index, row in enumerate(submission_rows)}

    manual_rows_applied = 0
    manual_rows_missing_in_base = 0
    for row_id, class_id in manual_labels.items():
        index = index_by_row_id.get(row_id)
        if index is None:
            manual_rows_missing_in_base += 1
            continue
        if int(submission_rows[index]["class_id"]) != class_id:
            manual_rows_applied += 1
        submission_rows[index]["class_id"] = str(class_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "class_id"])
        writer.writeheader()
        writer.writerows(submission_rows)

    return PatchSubmissionResult(
        total_rows=len(submission_rows),
        manual_rows_loaded=len(manual_labels),
        manual_rows_applied=manual_rows_applied,
        manual_rows_missing_in_base=manual_rows_missing_in_base,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = patch_submission(
        base_submission_path=args.base_submission,
        manual_labels_path=args.manual_labels,
        output_path=args.output,
        manual_source_name=args.manual_source,
    )

    print(
        f"Wrote {result.total_rows} rows to {args.output}. "
        f"Loaded {result.manual_rows_loaded} manual labels, "
        f"applied {result.manual_rows_applied}, "
        f"skipped {result.manual_rows_missing_in_base} unknown row_id values."
    )


if __name__ == "__main__":
    main()
