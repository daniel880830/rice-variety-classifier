"""Runtime settings (environment / .env) and experiment parameters (TOML)."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _default_root() -> Path:
    """The source checkout when running from it, otherwise the working directory.

    An installed (non-editable) package lives in site-packages, where ``parents[2]``
    is not the project, so e.g. the Docker image relies on its ``WORKDIR`` instead.
    """
    return _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()


PROJECT_ROOT = _default_root()


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


class Settings(BaseSettings):
    """Runtime settings read from environment variables prefixed with ``RICE_``.

    Relative paths are resolved against ``project_root`` so the CLI behaves the
    same regardless of the current working directory.
    """

    model_config = SettingsConfigDict(env_prefix="RICE_", env_file=".env", extra="ignore")

    project_root: Path = PROJECT_ROOT
    data_path: Path = Path("data/raw/rice_cammeo_osmancik.csv")
    params_path: Path = Path("config/params.toml")
    model_path: Path = Path("models/model.joblib")
    reports_dir: Path = Path("reports")

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    max_batch_size: int = Field(default=1000, ge=1)
    log_level: str = Field(default="info", pattern="^(critical|error|warning|info|debug)$")

    def resolve(self, path: Path) -> Path:
        """Return ``path`` as an absolute path anchored at the project root."""
        return path if path.is_absolute() else self.project_root / path


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()


def load_params(path: Path) -> dict[str, Any]:
    """Load and minimally validate the experiment parameter file.

    Args:
        path: Path to a TOML file such as ``config/params.toml``.

    Returns:
        The parsed parameters.

    Raises:
        ConfigError: If the file is missing, unparsable or has invalid values.
    """
    if not path.is_file():
        raise ConfigError(f"Parameter file not found: {path}")
    try:
        params = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    for section in ("data", "cv", "split", "bootstrap", "models"):
        if section not in params:
            raise ConfigError(f"Missing [{section}] section in {path}")
    if params["cv"]["n_splits"] < 2 or params["cv"]["n_repeats"] < 1:
        raise ConfigError("cv.n_splits must be >= 2 and cv.n_repeats >= 1")
    if not 0 < params["split"]["test_size"] < 1:
        raise ConfigError("split.test_size must be between 0 and 1")
    if not params["models"]:
        raise ConfigError(f"No [models.*] grids defined in {path}")
    for name, grid in params["models"].items():
        if not all(isinstance(v, list) and v for v in grid.values()):
            raise ConfigError(f"models.{name}: every hyper-parameter must be a non-empty list")
    return params
