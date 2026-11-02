"""Loading the model artifact and making predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from rice_classifier.schema import Prediction


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model artifact does not exist."""


class ModelLoadError(RuntimeError):
    """Raised when the model artifact exists but cannot be used."""


@dataclass(frozen=True)
class ModelBundle:
    """A fitted pipeline together with its training metadata."""

    pipeline: Pipeline
    metadata: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        """Feature columns in training order."""
        return list(self.metadata["feature_names"])

    @property
    def class_names(self) -> dict[int, str]:
        """Encoded label -> original class name."""
        return {v: k for k, v in self.metadata["label_map"].items()}


def load_model(path: Path) -> ModelBundle:
    """Load and sanity-check a model artifact written by ``train.save_model``.

    Args:
        path: Path to the ``.joblib`` artifact.

    Returns:
        The loaded bundle.

    Raises:
        ModelNotFoundError: If ``path`` does not exist.
        ModelLoadError: If the file is corrupt or not a valid artifact.
    """
    if not path.is_file():
        raise ModelNotFoundError(f"Model artifact not found: {path}. Run `rice train` first.")
    try:
        payload = joblib.load(path)
    except Exception as exc:  # joblib/pickle raise many unrelated exception types
        raise ModelLoadError(f"Could not load model artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or not {"pipeline", "metadata"} <= payload.keys():
        raise ModelLoadError(f"{path} is not a rice-classifier artifact")
    pipeline, metadata = payload["pipeline"], payload["metadata"]
    if not hasattr(pipeline, "predict_proba") or "feature_names" not in metadata:
        raise ModelLoadError(f"{path} is missing a pipeline or feature metadata")
    return ModelBundle(pipeline=pipeline, metadata=metadata)


def predict_records(
    bundle: ModelBundle, records: Sequence[Mapping[str, float | None]]
) -> list[Prediction]:
    """Classify raw-feature records.

    Records are aligned to the training column order, so the model always sees
    the same feature width and order as during training. ``None`` becomes NaN
    and is imputed by the pipeline.

    Args:
        bundle: Loaded model.
        records: One mapping per grain, keyed by raw column names.

    Returns:
        One prediction per record, in input order.

    Raises:
        ValueError: If ``records`` is empty or a record lacks a feature.
    """
    if not records:
        raise ValueError("No records to predict")
    features = bundle.feature_names
    for i, record in enumerate(records):
        missing = [f for f in features if f not in record]
        if missing:
            raise ValueError(f"Record {i} is missing features: {missing}")

    frame = pd.DataFrame([[r[f] for f in features] for r in records], columns=features, dtype=float)
    probabilities = bundle.pipeline.predict_proba(frame)
    classes = bundle.pipeline.classes_
    names = bundle.class_names
    results = []
    for row in probabilities:
        best = int(row.argmax())
        label = int(classes[best])
        results.append(
            Prediction(label=label, class_name=names[label], probability=float(row[best]))
        )
    return results
