import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rice_classifier.cli import app
from rice_classifier.config import get_settings

runner = CliRunner()


def test_train_writes_report_and_model(env: Path):
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0, result.output
    report = json.loads((env / "reports" / "metrics.json").read_text())
    assert report["selected_model"] in report["models"]
    assert (env / "out" / "model.joblib").is_file()
    assert "Selected by CV" in result.output


def test_train_subset_without_saving(env: Path):
    result = runner.invoke(app, ["train", "--model", "dummy", "--no-save"])
    assert result.exit_code == 0, result.output
    assert not (env / "out" / "model.joblib").exists()
    assert "logistic_regression" not in result.output


def test_train_missing_data(env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RICE_DATA_PATH", str(env / "nope.csv"))
    get_settings.cache_clear()
    result = runner.invoke(app, ["train"])
    assert isinstance(result.exception, FileNotFoundError)


def test_leakage_audit(env: Path):
    assert runner.invoke(app, ["train", "--no-save"]).exit_code == 0
    result = runner.invoke(app, ["leakage-audit"])
    assert result.exit_code == 0, result.output
    audit = json.loads((env / "reports" / "leakage_audit.json").read_text())
    assert "logistic_regression" in audit
    assert "gap" in result.output


def test_leakage_audit_requires_train(env: Path):
    result = runner.invoke(app, ["leakage-audit"])
    assert result.exit_code == 2
    assert "rice train" in result.output


def test_benchmark(env: Path):
    assert runner.invoke(app, ["train"]).exit_code == 0
    result = runner.invoke(app, ["benchmark"])
    assert result.exit_code == 0, result.output
    bench = json.loads((env / "reports" / "benchmark.json").read_text())
    assert bench["in_process"]["single_row"]["p95_ms"] > 0
    assert bench["artifact_kb"] > 0


def test_serve_uses_settings(env: Path, monkeypatch: pytest.MonkeyPatch):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(kw))
    monkeypatch.setenv("RICE_API_PORT", "8765")
    get_settings.cache_clear()
    assert runner.invoke(app, ["serve"]).exit_code == 0
    assert calls["port"] == 8765
    assert runner.invoke(app, ["serve", "--port", "9000", "--host", "0.0.0.0"]).exit_code == 0
    assert calls == {"host": "0.0.0.0", "port": 9000, "log_level": "warning"}
