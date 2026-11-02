"""Held-out metrics, bootstrap confidence intervals and the preprocessing-leakage audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.pipeline import Pipeline


def holdout_metrics(
    estimator: BaseEstimator,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Metrics of a fitted classifier on the held-out set.

    Args:
        estimator: Fitted estimator exposing ``predict`` and ``predict_proba``.
        X_test: Held-out features.
        y_test: Held-out labels.

    Returns:
        Accuracy, macro F1, Cohen's kappa, ROC AUC, Brier score and the confusion matrix.
    """
    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "brier": float(brier_score_loss(y_test, y_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_resamples: int,
    random_state: int,
    level: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for accuracy on a fixed test set.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        n_resamples: Number of bootstrap resamples.
        random_state: Seed for the resampling.
        level: Confidence level.

    Returns:
        ``low`` and ``high`` bounds of the interval.
    """
    rng = np.random.default_rng(random_state)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    idx = rng.integers(0, len(correct), size=(n_resamples, len(correct)))
    scores = correct[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return {"low": float(np.quantile(scores, alpha)), "high": float(np.quantile(scores, 1 - alpha))}


def leakage_gap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: BaseCrossValidator,
) -> dict[str, float]:
    """Compare CV accuracy with preprocessing fit inside vs. outside the folds.

    "Leaky" fits the pipeline's ``preprocess`` step on *all* rows first, so every
    validation fold has influenced the imputation and scaling statistics.

    Args:
        pipeline: Unfitted pipeline with ``preprocess`` and ``clf`` steps.
        X: Features.
        y: Labels.
        cv: Splitter (the same folds are used for both variants).

    Returns:
        Mean CV accuracy for both variants and their difference in percentage points.
    """
    clean = cross_val_score(clone(pipeline), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    X_leaky = clone(pipeline.named_steps["preprocess"]).fit_transform(X)
    leaky = cross_val_score(
        clone(pipeline.named_steps["clf"]), X_leaky, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return {
        "leak_free": float(clean.mean()),
        "leaky": float(leaky.mean()),
        "gap_pp": float(100 * (leaky.mean() - clean.mean())),
    }
