from pathlib import Path

import pytest
from pydantic import ValidationError

from rice_classifier.config import PROJECT_ROOT, ConfigError, Settings, load_params


def test_load_params_ok(params_path: Path):
    params = load_params(params_path)
    assert params["cv"]["n_splits"] == 3
    assert params["data"]["label_map"] == {"Cammeo": 0, "Osmancik": 1}


def test_project_params_file_is_valid():
    params = load_params(PROJECT_ROOT / "config" / "params.toml")
    assert params["split"]["test_size"] == 0.2


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_params(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text("[cv\nn_splits = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_params(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[split]", "[splitx]", r"Missing \[split\]"),
        ("n_splits = 3", "n_splits = 1", "n_splits"),
        ("test_size = 0.25", "test_size = 1.5", "test_size"),
        ("n_repeats = 2", "n_repeats = 0", "n_repeats"),
        ("[bootstrap]", "[bootstrapx]", r"Missing \[bootstrap\]"),
        ("C = [0.1, 1.0]", "C = []", "non-empty list"),
    ],
)
def test_invalid_values(params_path: Path, old: str, new: str, message: str):
    params_path.write_text(params_path.read_text().replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_params(params_path)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RICE_API_PORT", "9123")
    monkeypatch.setenv("RICE_MAX_BATCH_SIZE", "5")
    settings = Settings(_env_file=None)
    assert settings.api_port == 9123
    assert settings.max_batch_size == 5


@pytest.mark.parametrize(
    ("var", "value"),
    [("RICE_API_PORT", "70000"), ("RICE_MAX_BATCH_SIZE", "0"), ("RICE_LOG_LEVEL", "loud")],
)
def test_settings_reject_invalid(monkeypatch: pytest.MonkeyPatch, var: str, value: str):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_resolve_paths(tmp_path: Path):
    settings = Settings(_env_file=None)
    assert settings.resolve(Path("models/x")) == PROJECT_ROOT / "models" / "x"
    assert settings.resolve(tmp_path) == tmp_path


def test_installed_package_falls_back_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regression: inside site-packages there is no pyproject.toml next to the package."""
    from rice_classifier import config

    monkeypatch.setattr(config, "_SOURCE_ROOT", tmp_path / "site-packages")
    monkeypatch.chdir(tmp_path)
    assert config._default_root() == tmp_path


def test_project_root_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RICE_PROJECT_ROOT", str(tmp_path))
    assert Settings(_env_file=None).resolve(Path("x.csv")) == tmp_path / "x.csv"
