from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_csv_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items) or None


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    data_dir: Path
    state_dir: Path
    prediction_queue_path: Path | None
    kaggle_competition: str
    auto_download_dataset: bool
    annotate_sources: tuple[str, ...] | None
    include_labeled_sources: bool
    log_level: str

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required.")

        settings = cls(
            telegram_bot_token=token,
            data_dir=Path(os.getenv("DATA_DIR", ".")).expanduser().resolve(),
            state_dir=Path(os.getenv("STATE_DIR", ".state")).expanduser().resolve(),
            prediction_queue_path=(
                Path(os.getenv("PREDICTION_QUEUE_PATH", "")).expanduser().resolve()
                if os.getenv("PREDICTION_QUEUE_PATH")
                else None
            ),
            kaggle_competition=os.getenv(
                "KAGGLE_COMPETITION",
                "multi-view-pig-posture-recognition",
            ).strip(),
            auto_download_dataset=_parse_bool(
                os.getenv("AUTO_DOWNLOAD_DATASET"),
                default=False,
            ),
            annotate_sources=_parse_csv_list(os.getenv("ANNOTATE_SOURCES")),
            include_labeled_sources=_parse_bool(
                os.getenv("INCLUDE_LABELED_SOURCES"),
                default=False,
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        return settings

    def configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
