from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REQUIRED_TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"


class ConfigError(ValueError):
    """Raised when runtime configuration violates competition constraints."""


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    root: Path
    typhoon_api_key: str
    typhoon_base_url: str
    typhoon_model: str
    mock_typhoon: bool
    timeout_seconds: int
    max_retries: int
    cache_dir: Path
    outputs_dir: Path
    reports_dir: Path

    @property
    def use_mock_typhoon(self) -> bool:
        return self.mock_typhoon or not self.typhoon_api_key


def load_config(root: Path | None = None) -> AppConfig:
    repo_root = (root or Path(__file__).resolve().parents[1]).resolve()
    load_dotenv(repo_root / ".env")

    model = os.getenv("TYPHOON_MODEL", REQUIRED_TYPHOON_MODEL).strip()
    if model != REQUIRED_TYPHOON_MODEL:
        raise ConfigError(
            f"TYPHOON_MODEL must be {REQUIRED_TYPHOON_MODEL!r}; got {model!r}"
        )

    timeout_raw = os.getenv("TYPHOON_TIMEOUT_SECONDS", "60").strip()
    retries_raw = os.getenv("TYPHOON_MAX_RETRIES", "2").strip()
    try:
        timeout = int(timeout_raw)
        retries = int(retries_raw)
    except ValueError as exc:
        raise ConfigError("TYPHOON_TIMEOUT_SECONDS and TYPHOON_MAX_RETRIES must be integers") from exc

    if timeout <= 0:
        raise ConfigError("TYPHOON_TIMEOUT_SECONDS must be positive")
    if retries < 0 or retries > 2:
        raise ConfigError("TYPHOON_MAX_RETRIES must be between 0 and 2")

    return AppConfig(
        root=repo_root,
        typhoon_api_key=os.getenv("TYPHOON_API_KEY", "").strip(),
        typhoon_base_url=os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1").rstrip("/"),
        typhoon_model=model,
        mock_typhoon=_bool_env(os.getenv("MOCK_TYPHOON"), False),
        timeout_seconds=timeout,
        max_retries=retries,
        cache_dir=repo_root / "cache",
        outputs_dir=repo_root / "outputs",
        reports_dir=repo_root / "reports",
    )


def ensure_runtime_dirs(config: AppConfig) -> None:
    for directory in [config.cache_dir, config.outputs_dir, config.reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

