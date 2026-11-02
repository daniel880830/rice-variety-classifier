"""Checks on the bundled public dataset (UCI Rice, Cammeo and Osmancik)."""

import pytest

from rice_classifier.config import PROJECT_ROOT, load_params
from rice_classifier.data import load_dataset
from rice_classifier.schema import RAW_FEATURES
from rice_classifier.train import run_experiment

DATA = PROJECT_ROOT / "data" / "raw" / "rice_cammeo_osmancik.csv"
# SHA-256 of the CSV converted from the UCI ARFF file (see data/README.md).
DATA_SHA256 = "307084f7903e863f55737d9878c08cd6b017fceb0699801d1cca05439541c56f"


@pytest.fixture(scope="module")
def dataset():
    params = load_params(PROJECT_ROOT / "config" / "params.toml")
    return load_dataset(
        DATA, target="Class", label_map=params["data"]["label_map"], expected_features=RAW_FEATURES
    )


def test_dataset_matches_published_description(dataset):
    assert len(dataset.y) == 3810
    assert dataset.y.tolist().count(0) == 1630  # Cammeo
    assert dataset.y.tolist().count(1) == 2180  # Osmancik
    assert dataset.X.isna().sum().sum() == 0
    assert dataset.sha256 == DATA_SHA256


@pytest.mark.slow
def test_logistic_regression_result_is_stable(dataset):
    params = load_params(PROJECT_ROOT / "config" / "params.toml")
    result = run_experiment(dataset, params, models=["logistic_regression"]).report
    lr = result["models"]["logistic_regression"]
    assert lr["best_params"] == {"C": 10.0}
    assert round(lr["cv_accuracy"]["mean"], 4) == 0.9339
    assert round(lr["test"]["accuracy"], 4) == 0.9199
