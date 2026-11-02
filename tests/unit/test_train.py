import json

import joblib
import numpy as np
import pytest

from rice_classifier.train import (
    run_experiment,
    run_leakage_audit,
    save_json,
    save_model,
    split_data,
)


def test_experiment_report_structure(dataset, params):
    result = run_experiment(dataset, params)
    report = result.report
    assert set(report["models"]) == {"dummy", "logistic_regression", "svm_rbf", "random_forest"}
    assert report["split"] == {"train_rows": 90, "test_rows": 30}
    assert report["cv"]["folds"] == 6
    assert report["selected_model"] in report["models"]
    ci = report["models"]["logistic_regression"]["test"]["accuracy_ci95"]
    assert ci["low"] <= report["models"]["logistic_regression"]["test"]["accuracy"] <= ci["high"]
    assert result.metadata["feature_names"] == list(dataset.feature_names)
    assert hasattr(result.model, "predict_proba")


def test_selection_uses_cv_not_test(dataset, params):
    report = run_experiment(dataset, params).report
    best_cv = max(r["cv_accuracy"]["mean"] for r in report["models"].values())
    assert report["models"][report["selected_model"]]["cv_accuracy"]["mean"] == best_cv


def test_model_subset_and_unknown(dataset, params):
    report = run_experiment(dataset, params, models=["dummy"]).report
    assert list(report["models"]) == ["dummy"]
    with pytest.raises(KeyError, match="No \\[models"):
        run_experiment(dataset, params, models=["xgboost"])


def test_wrapped_svm_params_round_trip(dataset, params):
    best = run_experiment(dataset, params, models=["svm_rbf"]).report["models"]["svm_rbf"]
    assert best["best_params"] == {"C": 1.0, "gamma": "scale"}


def test_results_are_reproducible(dataset, params):
    assert run_experiment(dataset, params).report == run_experiment(dataset, params).report


def test_train_and_test_rows_do_not_overlap(dataset, params):
    split = split_data(dataset, params)
    assert not set(split.X_train.index) & set(split.X_test.index)
    assert len(split.X_train) + len(split.X_test) == len(dataset.y)
    assert np.bincount(split.y_test).tolist() == [15, 15]


def test_served_model_never_sees_test_rows(dataset, params):
    """The selected pipeline's imputer statistics come from the training split only."""
    model = run_experiment(dataset, params).model
    split = split_data(dataset, params)
    imputer = model.named_steps["preprocess"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, split.X_train.median().to_numpy())


def test_leakage_audit(dataset, params):
    report = run_experiment(dataset, params).report
    audit = run_leakage_audit(dataset, report, params)
    assert set(audit) == set(report["models"])
    row = audit["svm_rbf"]
    assert row["gap_pp"] == pytest.approx(100 * (row["leaky"] - row["leak_free"]))


def test_save_json_and_model(tmp_path, dataset, params):
    result = run_experiment(dataset, params, models=["dummy"])
    path = save_json(result.report, tmp_path / "r" / "metrics.json")
    assert json.loads(path.read_text())["selected_model"] == "dummy"
    model_path = save_model(result.model, result.metadata, tmp_path / "m" / "model.joblib")
    assert joblib.load(model_path)["metadata"]["model_name"] == "dummy"
