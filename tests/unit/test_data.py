from pathlib import Path

import numpy as np
import pytest

from rice_classifier.data import DataValidationError, load_dataset
from rice_classifier.schema import RAW_FEATURES
from tests.conftest import make_rice_frame

LABELS = {"Cammeo": 0, "Osmancik": 1}


def _write(tmp_path: Path, frame) -> Path:
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_converts_missing_and_labels(rice_csv: Path):
    ds = load_dataset(rice_csv, target="Class", label_map=LABELS, expected_features=RAW_FEATURES)
    assert ds.X.shape == (120, 7)
    assert ds.feature_names == RAW_FEATURES
    assert ds.X.isna().sum().sum() == 6
    assert all(dtype == np.float64 for dtype in ds.X.dtypes)
    assert set(ds.y) == {0, 1}
    assert len(ds.sha256) == 64


def test_infers_features_when_not_given(tmp_path: Path):
    frame = make_rice_frame().drop(columns=["Extent"])
    ds = load_dataset(_write(tmp_path, frame), target="Class", label_map=LABELS)
    assert ds.feature_names == RAW_FEATURES[:-1]


def test_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv", target="Class", label_map=LABELS)


def test_empty_file(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="empty"):
        load_dataset(path, target="Class", label_map=LABELS)


def test_header_only(tmp_path: Path):
    path = tmp_path / "header.csv"
    path.write_text(",".join([*RAW_FEATURES, "Class"]) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="no rows"):
        load_dataset(path, target="Class", label_map=LABELS)


def test_missing_target(tmp_path: Path):
    path = _write(tmp_path, make_rice_frame().drop(columns=["Class"]))
    with pytest.raises(DataValidationError, match="Target column"):
        load_dataset(path, target="Class", label_map=LABELS)


def test_missing_feature_column(tmp_path: Path):
    path = _write(tmp_path, make_rice_frame().drop(columns=["Area"]))
    with pytest.raises(DataValidationError, match="Missing feature columns"):
        load_dataset(path, target="Class", label_map=LABELS, expected_features=RAW_FEATURES)


def test_non_numeric_value(tmp_path: Path):
    frame = make_rice_frame()
    frame.loc[3, "Extent"] = "abc"
    with pytest.raises(DataValidationError, match="Non-numeric value 'abc' in column 'Extent'"):
        load_dataset(_write(tmp_path, frame), target="Class", label_map=LABELS)


def test_unknown_label(tmp_path: Path):
    frame = make_rice_frame()
    frame.loc[0, "Class"] = "Basmati"
    with pytest.raises(DataValidationError, match="Basmati"):
        load_dataset(_write(tmp_path, frame), target="Class", label_map=LABELS)


def test_missing_label(tmp_path: Path):
    frame = make_rice_frame()
    frame.loc[0, "Class"] = None
    with pytest.raises(DataValidationError, match="NaN"):
        load_dataset(_write(tmp_path, frame), target="Class", label_map=LABELS)
