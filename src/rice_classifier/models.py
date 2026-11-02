"""Candidate classifiers searched by ``rice train``."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

SEED = 42

FACTORIES: dict[str, Callable[[], ClassifierMixin]] = {
    "dummy": DummyClassifier,
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
    "knn": KNeighborsClassifier,
    # Platt-scaled probabilities for the API; replaces the deprecated SVC(probability=True).
    "svm_rbf": lambda: CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False),
    "random_forest": lambda: RandomForestClassifier(n_jobs=1, random_state=SEED),
    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}


# Searched parameters of wrapped estimators live under a nested prefix.
PARAM_PREFIX: dict[str, str] = {"svm_rbf": "estimator__"}


def param_names(name: str, params: dict[str, object]) -> dict[str, object]:
    """Map ``params.toml`` keys to the estimator's ``set_params`` names."""
    prefix = PARAM_PREFIX.get(name, "")
    return {f"{prefix}{k}": v for k, v in params.items()}


def build_model(name: str) -> ClassifierMixin:
    """Return an unfitted candidate classifier.

    Args:
        name: Key of a ``[models.<name>]`` table in ``params.toml``.

    Returns:
        The classifier with its fixed (non-searched) settings applied.

    Raises:
        KeyError: If ``name`` is not a known model.
    """
    if name not in FACTORIES:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(FACTORIES)}")
    return FACTORIES[name]()
