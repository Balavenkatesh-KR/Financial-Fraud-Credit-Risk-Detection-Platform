from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KaggleIngestor:
    """Downloads real datasets using Kaggle CLI."""

    def __init__(self, raw_data_dir: str = "data/raw") -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def download_competition(self, competition: str) -> Path:
        target_dir = self.raw_data_dir / competition
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(target_dir)]
        logger.info("Downloading competition dataset: %s", competition)
        subprocess.run(cmd, check=True)
        return target_dir

    def download_dataset(self, dataset: str) -> Path:
        slug = dataset.replace("/", "_")
        target_dir = self.raw_data_dir / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target_dir)]
        logger.info("Downloading dataset: %s", dataset)
        subprocess.run(cmd, check=True)
        return target_dir
