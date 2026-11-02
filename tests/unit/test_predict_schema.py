from pathlib import Path

import joblib
import pytest
from pydantic import ValidationError

from rice_classifier.predict import ModelLoadError, ModelNotFoundError, load_model, predict_records
from rice_classifier.schema import BatchRequest, RiceGrain


def test_load_model(model_path: Path):
    bundle = load_model(model_path)
    assert bundle.metadata["model_name"] in {
        "dummy",
        "logistic_regression",
        "svm_rbf",
        "random_forest",
    }
    assert bundle.class_names == {0: "Cammeo", 1: "Osmancik"}


def test_load_model_missing(tmp_path: Path):
    with pytest.raises(ModelNotFoundError, match="rice train"):
        load_model(tmp_path / "none.joblib")


def test_load_model_corrupt(tmp_path: Path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"not a pickle at all")
    with pytest.raises(ModelLoadError, match="Could not load"):
        load_model(path)


@pytest.mark.parametrize(
    "payload",
    [{"pipeline": object()}, ["a", "list"], {"pipeline": object(), "metadata": {}}],
)
def test_load_model_wrong_payload(tmp_path: Path, payload):
    path = tmp_path / "wrong.joblib"
    joblib.dump(payload, path)
    with pytest.raises(ModelLoadError):
        load_model(path)


def test_single_prediction_matches_batch(model_path: Path, dataset):
    bundle = load_model(model_path)
    records = dataset.X.head(20).astype(object).where(dataset.X.head(20).notna(), None)
    records = records.to_dict(orient="records")
    batch = predict_records(bundle, records)
    singles = [predict_records(bundle, [r])[0] for r in records]
    assert batch == singles
    assert all(0.5 <= p.probability <= 1 for p in batch)


def test_feature_order_does_not_matter(model_path: Path, grain: dict):
    bundle = load_model(model_path)
    raw = RiceGrain(**grain).to_raw()
    reversed_raw = dict(reversed(list(raw.items())))
    assert predict_records(bundle, [raw]) == predict_records(bundle, [reversed_raw])


def test_single_row_has_training_feature_width(model_path: Path, grain: dict):
    bundle = load_model(model_path)
    preprocess = bundle.pipeline.named_steps["preprocess"]
    assert preprocess.n_features_in_ == len(bundle.feature_names) == 7
    predict_records(bundle, [RiceGrain(**grain).to_raw()])


def test_missing_value_is_imputed(model_path: Path, grain: dict):
    bundle = load_model(model_path)
    prediction = predict_records(bundle, [RiceGrain(**{**grain, "area": None}).to_raw()])[0]
    assert prediction.class_name in {"Cammeo", "Osmancik"}


def test_predict_empty(model_path: Path):
    with pytest.raises(ValueError, match="No records"):
        predict_records(load_model(model_path), [])


def test_predict_missing_feature(model_path: Path, grain: dict):
    raw = RiceGrain(**grain).to_raw()
    raw.pop("Extent")
    with pytest.raises(ValueError, match="missing features"):
        predict_records(load_model(model_path), [raw])


def test_schema_maps_to_raw_columns(grain: dict):
    raw = RiceGrain(**grain).to_raw()
    assert raw["Perimeter"] == grain["perimeter"]
    assert len(raw) == 7


@pytest.mark.parametrize(
    "change",
    [{"unexpected": 1}, {"eccentricity": 1.5}, {"area": -3}, {"extent": "wide"}],
)
def test_schema_rejects_bad_input(grain: dict, change: dict):
    with pytest.raises(ValidationError):
        RiceGrain(**{**grain, **change})


def test_schema_requires_every_field(grain: dict):
    grain.pop("area")
    with pytest.raises(ValidationError):
        RiceGrain(**grain)


def test_batch_rejects_empty():
    with pytest.raises(ValidationError):
        BatchRequest(items=[])
