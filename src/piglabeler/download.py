from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from piglabeler.constants import REQUIRED_DATA_COLUMNS
from piglabeler.config import Settings

LOGGER = logging.getLogger(__name__)


def ensure_dataset_ready(settings: Settings) -> None:
    if _has_usable_dataset(settings.data_dir):
        LOGGER.info("Dataset is already present in %s", settings.data_dir)
        return

    if not settings.auto_download_dataset:
        raise RuntimeError(
            f"No usable dataset found in {settings.data_dir}. "
            "Either mount the dataset there or enable AUTO_DOWNLOAD_DATASET=true."
        )

    LOGGER.info("Dataset not found. Downloading Kaggle competition files into %s", settings.data_dir)
    env = os.environ.copy()
    _prepare_kaggle_auth(settings, env)
    kaggle_executable = shutil.which("kaggle", path=env.get("PATH"))
    if not kaggle_executable:
        raise RuntimeError("Kaggle CLI executable was not found in PATH inside the container.")

    command = [
        kaggle_executable,
        "competitions",
        "download",
        "-c",
        settings.kaggle_competition,
        "-p",
        str(settings.data_dir),
        "--force",
    ]
    completed = subprocess.run(command, check=False, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "Failed to download the dataset from Kaggle. "
            "Make sure credentials are correct and the competition rules were accepted in the browser. "
            f"returncode={completed.returncode}; stdout={stdout!r}; stderr={stderr!r}"
        )

    _extract_all_archives(settings.data_dir)
    if not _has_usable_dataset(settings.data_dir):
        raise RuntimeError(
            f"Kaggle download finished, but expected CSV/image files were not found in {settings.data_dir}."
        )


def _prepare_kaggle_auth(settings: Settings, env: dict[str, str]) -> None:
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return

    home_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if home_kaggle_json.exists():
        return

    config_dir = settings.state_dir / ".kaggle"
    kaggle_json = config_dir / "kaggle.json"
    if kaggle_json.exists():
        env["KAGGLE_CONFIG_DIR"] = str(config_dir)
        return

    raise RuntimeError(
        "Kaggle credentials are missing. Set KAGGLE_USERNAME and KAGGLE_KEY in the environment, "
        "or provide kaggle.json in ~/.kaggle or STATE_DIR/.kaggle."
    )


def _extract_all_archives(data_dir: Path) -> None:
    for archive in sorted(data_dir.glob("*.zip")):
        LOGGER.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(data_dir)


def _has_usable_dataset(data_dir: Path) -> bool:
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.stem == "sample_submission":
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            if all(column in reader.fieldnames for column in REQUIRED_DATA_COLUMNS):
                image_dir = data_dir / f"{csv_path.stem}_images"
                if image_dir.exists():
                    return True
    return False
