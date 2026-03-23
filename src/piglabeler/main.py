from __future__ import annotations

import logging

from piglabeler.bot import PigLabelerBot
from piglabeler.config import Settings
from piglabeler.dataset import DatasetCatalog
from piglabeler.download import ensure_dataset_ready
from piglabeler.help_assets import HelpAssetBuilder
from piglabeler.rendering import TaskRenderer
from piglabeler.store import AnnotationStore


def main() -> None:
    settings = Settings.load()
    settings.configure_logging()

    logging.getLogger(__name__).info("Starting pig labeler bot")
    dataset_dir = ensure_dataset_ready(settings)

    catalog = DatasetCatalog.discover(
        dataset_dir,
        annotate_sources=settings.annotate_sources,
        include_labeled_sources=settings.include_labeled_sources,
    )
    if not catalog.annotatable_row_ids:
        logging.getLogger(__name__).warning(
            "No annotatable rows were discovered. The bot will start in read-only mode."
        )

    bot = PigLabelerBot(
        token=settings.telegram_bot_token,
        catalog=catalog,
        store=AnnotationStore(settings.state_dir),
        renderer=TaskRenderer(settings.state_dir),
        help_builder=HelpAssetBuilder(settings.state_dir),
    )
    bot.run()


if __name__ == "__main__":
    main()
