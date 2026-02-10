from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile


def unzip_all(directory: str | Path) -> None:
    directory = Path(directory)
    for zip_path in directory.rglob("*.zip"):
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)
