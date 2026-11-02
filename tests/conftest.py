from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rice_classifier.config import get_settings, load_params
from rice_classifier.data import Dataset, load_dataset
from rice_classifier.schema import RAW_FEATURES
from rice_classifier.train import run_experiment, save_model

# Small grids and 3 folds keep a full experiment to a few seconds.
FAST_PARAMS = """
[data]
target = "Class"
label_map = { Cammeo = 0, Osmancik = 1 }

[split]
test_size = 0.25
random_state = 0

[cv]
n_splits = 3
n_repeats = 2
random_state = 0

[bootstrap]
n_resamples = 200
random_state = 0

[models.dummy]
strategy = ["most_frequent"]

[models.logistic_regression]
C = [0.1, 1.0]

[models.svm_rbf]
C = [1.0]
gamma = ["scale"]

[models.random_forest]
n_estimators = [10]
max_depth = [3]
"""


def make_rice_frame(n: int = 120, seed: int = 0, missing: int = 6) -> pd.DataFrame:
    """Synthetic grains: Cammeo is larger and more elongated than Osmancik."""
    rng = np.random.default_rng(seed)
    labels = np.array(["Cammeo", "Osmancik"] * (n // 2))
    big = labels == "Cammeo"
    area = np.where(big, 14_000, 11_500) + rng.normal(0, 900, n)
    major = np.where(big, 200, 175) + rng.normal(0, 8, n)
    minor = np.where(big, 89, 84) + rng.normal(0, 4, n)
    frame = pd.DataFrame(
        {
            "Area": area.round(),
            "Perimeter": np.where(big, 480, 430) + rng.normal(0, 15, n),
            "Major_Axis_Length": major,
            "Minor_Axis_Length": minor,
            "Eccentricity": np.clip(np.sqrt(1 - (minor / major) ** 2), 0, 1),
            "Convex_Area": (area * 1.02).round(),
            "Extent": rng.uniform(0.5, 0.85, n),
            "Class": labels,
        }
    ).astype({c: object for c in RAW_FEATURES})
    for i in range(missing):
        frame.iat[(i * 17) % n, i % len(RAW_FEATURES)] = np.nan
    return frame


@pytest.fixture
def rice_csv(tmp_path: Path) -> Path:
    path = tmp_path / "rice.csv"
    make_rice_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def params_path(tmp_path: Path) -> Path:
    path = tmp_path / "params.toml"
    path.write_text(FAST_PARAMS, encoding="utf-8")
    return path


@pytest.fixture
def params(params_path: Path) -> dict:
    return load_params(params_path)


@pytest.fixture
def dataset(rice_csv: Path, params: dict) -> Dataset:
    return load_dataset(
        rice_csv,
        target="Class",
        label_map=params["data"]["label_map"],
        expected_features=RAW_FEATURES,
    )


@pytest.fixture
def model_path(tmp_path: Path, dataset: Dataset, params: dict) -> Path:
    result = run_experiment(dataset, params)
    return save_model(result.model, result.metadata, tmp_path / "model.joblib")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rice_csv: Path, params_path: Path):
    """Point every RICE_* path at temporary files and reset the settings cache."""
    monkeypatch.setenv("RICE_DATA_PATH", str(rice_csv))
    monkeypatch.setenv("RICE_PARAMS_PATH", str(params_path))
    monkeypatch.setenv("RICE_MODEL_PATH", str(tmp_path / "out" / "model.joblib"))
    monkeypatch.setenv("RICE_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("RICE_LOG_LEVEL", "warning")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def grain() -> dict:
    return {
        "area": 15231,
        "perimeter": 525.58,
        "major_axis_length": 229.75,
        "minor_axis_length": 85.09,
        "eccentricity": 0.9289,
        "convex_area": 15617,
        "extent": 0.5729,
    }
