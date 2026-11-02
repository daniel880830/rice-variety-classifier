"""Preprocessing shared by training and inference."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    """Median imputation followed by standardisation (zero mean, unit variance).

    Returns:
        An unfitted preprocessing pipeline.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def build_pipeline(classifier: ClassifierMixin) -> Pipeline:
    """Chain the preprocessor and a classifier into one estimator.

    Fitting the whole pipeline inside each CV fold keeps imputation and scaling
    statistics from ever seeing validation or test rows, and the saved
    artifact applies the identical transformation at inference time.

    Args:
        classifier: An unfitted sklearn classifier.

    Returns:
        A pipeline with steps ``preprocess`` and ``clf``.
    """
    return Pipeline([("preprocess", build_preprocessor()), ("clf", classifier)])
