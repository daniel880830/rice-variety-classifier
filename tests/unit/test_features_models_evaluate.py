import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from rice_classifier.evaluate import bootstrap_accuracy_ci, holdout_metrics, leakage_gap
from rice_classifier.features import build_pipeline, build_preprocessor
from rice_classifier.models import FACTORIES, build_model, param_names


def test_preprocessor_imputes_and_standardises():
    X = pd.DataFrame({"a": [0.0, 10.0, np.nan, 20.0], "b": [5.0, 5.0, 7.0, 9.0]})
    out = build_preprocessor().fit_transform(X)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0), 1)


def test_pipeline_learns_statistics_from_training_rows_only(dataset):
    X_train = dataset.X.iloc[:80]
    pipe = build_pipeline(DummyClassifier()).fit(X_train, dataset.y[:80])
    imputer = pipe.named_steps["preprocess"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, X_train.median().to_numpy())


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_every_model_builds_and_predicts_proba(name, dataset):
    pipe = build_pipeline(build_model(name)).fit(dataset.X, dataset.y)
    proba = pipe.predict_proba(dataset.X.head(3))
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_unknown_model():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("xgboost")


def test_param_names_prefix_wrapped_estimators():
    assert param_names("svm_rbf", {"C": 1}) == {"estimator__C": 1}
    assert param_names("knn", {"n_neighbors": 5}) == {"n_neighbors": 5}


def test_holdout_metrics_perfect_classifier():
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    clf = build_pipeline(build_model("knn").set_params(n_neighbors=1)).fit(X, y)
    metrics = holdout_metrics(clf, X, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier"] == 0.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_bootstrap_ci():
    y = np.array([0, 1] * 50)
    perfect = bootstrap_accuracy_ci(y, y, n_resamples=200, random_state=0)
    assert perfect == {"low": 1.0, "high": 1.0}
    noisy = y.copy()
    noisy[:20] = 1 - noisy[:20]
    ci = bootstrap_accuracy_ci(y, noisy, n_resamples=500, random_state=0)
    assert ci["low"] < 0.8 < ci["high"]


def test_leakage_gap_detects_scale_leak():
    """A test fold with a shifted scale makes the leaky variant look different."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": np.r_[rng.normal(0, 1, 50), rng.normal(0, 1, 50) * 100]})
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    result = leakage_gap(build_pipeline(build_model("knn")), X, y, folds)
    assert set(result) == {"leak_free", "leaky", "gap_pp"}
    assert result["gap_pp"] == pytest.approx(100 * (result["leaky"] - result["leak_free"]))
