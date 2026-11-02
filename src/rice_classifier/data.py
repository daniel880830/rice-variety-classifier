"""Loading and validating the raw CSV data."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input dataset does not match the expected format."""


@dataclass(frozen=True)
class Dataset:
    """Features and encoded labels.

    Attributes:
        X: Numeric feature frame; missing values are ``NaN``.
        y: Integer-encoded labels.
        feature_names: Column order of ``X``.
        sha256: Hash of the source file, recorded in model metadata.
    """

    X: pd.DataFrame
    y: np.ndarray
    feature_names: tuple[str, ...]
    sha256: str


def load_dataset(
    path: Path,
    *,
    target: str,
    label_map: Mapping[str, int],
    missing_token: str = "?",
    expected_features: Sequence[str] | None = None,
) -> Dataset:
    """Read a CSV, convert missing tokens to NaN and encode the target.

    Args:
        path: CSV file with feature columns and a target column.
        target: Name of the target column.
        label_map: Mapping from raw class names to integer labels.
        missing_token: String used for missing values in the source file.
        expected_features: If given, the exact feature columns required, in order.
            Otherwise every non-target column is used.

    Returns:
        The validated dataset.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If the file is empty, columns are missing, a feature is
            not numeric, or a label is not in ``label_map``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw_bytes = path.read_bytes()
    if not raw_bytes.strip():
        raise DataValidationError(f"Data file is empty: {path}")

    try:
        frame = pd.read_csv(path, na_values=[missing_token], keep_default_na=True)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Data file has no columns: {path}") from exc
    if frame.empty:
        raise DataValidationError(f"Data file has a header but no rows: {path}")
    if target not in frame.columns:
        raise DataValidationError(f"Target column '{target}' not found in {path}")

    features = (
        list(expected_features)
        if expected_features is not None
        else [c for c in frame.columns if c != target]
    )
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing feature columns: {missing}")

    X = frame[features].apply(pd.to_numeric, errors="coerce")
    bad = X.isna() & frame[features].notna()
    if bad.any().any():
        column = bad.any()[bad.any()].index[0]
        value = frame.loc[bad[column], column].iloc[0]
        raise DataValidationError(f"Non-numeric value {value!r} in column '{column}'")

    labels = frame[target]
    unknown = sorted(set(labels.dropna().astype(str)) - set(label_map))
    if unknown or labels.isna().any():
        raise DataValidationError(f"Unknown or missing labels: {unknown or ['<NaN>']}")
    y = labels.astype(str).map(label_map).to_numpy(dtype=int)

    return Dataset(
        X=X.astype(float),
        y=y,
        feature_names=tuple(features),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
