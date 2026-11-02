"""Model selection on the training split, one evaluation on the test split."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from rice_classifier.data import Dataset
from rice_classifier.evaluate import bootstrap_accuracy_ci, holdout_metrics, leakage_gap
from rice_classifier.features import build_pipeline
from rice_classifier.models import PARAM_PREFIX, build_model, param_names

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Outcome of ``run_experiment``.

    Attributes:
        report: JSON-serialisable results for every candidate model.
        model: The selected pipeline, refit on the whole training split.
        metadata: Provenance stored alongside the model artifact.
    """

    report: dict[str, Any]
    model: Pipeline
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Split:
    """Stratified train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def split_data(dataset: Dataset, params: dict[str, Any]) -> Split:
    """Stratified train/test split configured by ``[split]``."""
    cfg = params["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=cfg["test_size"],
        stratify=dataset.y,
        random_state=cfg["random_state"],
    )
    return Split(X_train, X_test, y_train, y_test)


def repeated_cv(params: dict[str, Any]) -> RepeatedStratifiedKFold:
    """The repeated stratified K-fold splitter configured by ``[cv]``."""
    cfg = params["cv"]
    return RepeatedStratifiedKFold(
        n_splits=cfg["n_splits"], n_repeats=cfg["n_repeats"], random_state=cfg["random_state"]
    )


def run_experiment(
    dataset: Dataset, params: dict[str, Any], *, models: list[str] | None = None
) -> ExperimentResult:
    """Grid-search every candidate on the training split and pick the best by CV.

    Each candidate is a pipeline (imputer → scaler → classifier), so preprocessing is
    re-fit inside every CV fold. The test split is used only to report the metrics of
    each tuned candidate; selection never looks at it.

    Args:
        dataset: Loaded dataset.
        params: Parsed ``params.toml``.
        models: Subset of ``[models.*]`` names to run (default: all).

    Returns:
        Per-model results, the selected refit pipeline and its metadata.

    Raises:
        KeyError: If a requested model has no grid in ``params``.
    """
    names = models or list(params["models"])
    missing = [n for n in names if n not in params["models"]]
    if missing:
        raise KeyError(f"No [models.*] grid for {missing}")

    split = split_data(dataset, params)
    cv = repeated_cv(params)
    boot = params["bootstrap"]
    results: dict[str, Any] = {}
    searches: dict[str, GridSearchCV] = {}
    for name in names:
        logger.info("grid search | %s", name)
        grid = {f"clf__{k}": v for k, v in param_names(name, params["models"][name]).items()}
        search = GridSearchCV(
            build_pipeline(build_model(name)), grid, cv=cv, scoring="accuracy", n_jobs=-1
        ).fit(split.X_train, split.y_train)
        searches[name] = search
        best = search.best_index_
        test = holdout_metrics(search.best_estimator_, split.X_test, split.y_test)
        test["accuracy_ci95"] = bootstrap_accuracy_ci(
            split.y_test,
            search.best_estimator_.predict(split.X_test),
            n_resamples=boot["n_resamples"],
            random_state=boot["random_state"],
        )
        results[name] = {
            "best_params": {
                k.removeprefix("clf__").removeprefix(PARAM_PREFIX.get(name, "")): v
                for k, v in search.best_params_.items()
            },
            "cv_accuracy": {
                "mean": float(search.cv_results_["mean_test_score"][best]),
                "std": float(search.cv_results_["std_test_score"][best]),
                "folds": cv.get_n_splits(),
            },
            "test": test,
        }

    selected = max(results, key=lambda n: results[n]["cv_accuracy"]["mean"])
    report = {
        "data": {
            "rows": len(dataset.y),
            "features": list(dataset.feature_names),
            "class_counts": np.bincount(dataset.y).tolist(),
            "sha256": dataset.sha256,
        },
        "split": {"train_rows": len(split.y_train), "test_rows": len(split.y_test)},
        "cv": {**params["cv"], "folds": cv.get_n_splits()},
        "sklearn_version": sklearn.__version__,
        "models": results,
        "selected_model": selected,
    }
    metadata = {
        "model_name": selected,
        "best_params": results[selected]["best_params"],
        "cv_accuracy": results[selected]["cv_accuracy"],
        "test_metrics": {
            k: v for k, v in results[selected]["test"].items() if k != "confusion_matrix"
        },
        "feature_names": list(dataset.feature_names),
        "label_map": dict(params["data"]["label_map"]),
        "data_sha256": dataset.sha256,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return ExperimentResult(
        report=report, model=searches[selected].best_estimator_, metadata=metadata
    )


def run_leakage_audit(dataset: Dataset, report: dict[str, Any], params: dict[str, Any]) -> dict:
    """Measure how much fitting preprocessing on all rows would inflate CV accuracy.

    Uses each model's tuned hyper-parameters from a ``train`` report and the training
    split only, with identical folds for the leaky and leak-free variants.

    Args:
        dataset: Loaded dataset.
        report: Output of ``run_experiment`` (``models.<name>.best_params``).
        params: Parsed ``params.toml``.

    Returns:
        ``{model: {leak_free, leaky, gap_pp}}``.
    """
    split = split_data(dataset, params)
    cfg = params["cv"]
    folds = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["random_state"]
    )
    audit = {}
    for name, result in report["models"].items():
        clf = build_model(name).set_params(**param_names(name, result["best_params"]))
        pipeline = build_pipeline(clf)
        audit[name] = leakage_gap(pipeline, split.X_train, split.y_train, folds)
    return audit


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Write ``payload`` as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def save_model(model: Pipeline, metadata: dict[str, Any], path: Path) -> Path:
    """Persist the pipeline and its metadata as a single joblib artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "metadata": metadata}, path)
    return path
