from pathlib import Path

import joblib


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_joblib(obj, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    joblib.dump(obj, path)


def load_joblib(path: str | Path):
    return joblib.load(path)
