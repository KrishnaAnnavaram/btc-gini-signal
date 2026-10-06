"""Settings from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """A setting has an invalid value."""


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                value = value.strip().strip("'\"")
                if value:
                    out[key.strip()] = value
    return out


def _int(env: Mapping[str, str], key: str, default: int, minimum: int) -> int:
    raw = env.get(key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float, minimum: float) -> float:
    raw = env.get(key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


def parse_horizons(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(p) for p in raw.split(",") if p.strip()}))
    except ValueError as exc:
        raise ConfigError(f"horizons must be integers, got {raw!r}") from exc
    if not values or values[0] < 1:
        raise ConfigError(f"horizons must be >= 1 day (a same-day target is not a forecast), got {raw!r}")
    return values


@dataclass(frozen=True)
class Settings:
    data_path: Path | None = None
    output_dir: Path = Path("outputs")
    processed_dir: Path = Path("data/processed")
    seed: int = 42
    horizons: tuple[int, ...] = (1, 7)
    window: int = 30
    min_train: int = 730
    refit_every: int = 30
    cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if not self.horizons or min(self.horizons) < 1:
            raise ConfigError("horizons must be >= 1 day")
        if self.window < 5:
            raise ConfigError("the inequality window must be >= 5 days")
        if self.min_train < 50 or self.refit_every < 1:
            raise ConfigError("min_train must be >= 50 and refit_every >= 1")

    def with_overrides(self, **changes) -> "Settings":
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: Path | None = Path(".env")) -> "Settings":
        merged: dict[str, str] = {}
        if env is None:
            if dotenv is not None:
                merged.update(_read_dotenv(dotenv))
            merged.update(os.environ)
        else:
            merged.update(env)
        data = merged.get("BTCGINI_DATA")
        horizons = merged.get("BTCGINI_HORIZONS")
        return cls(
            data_path=Path(data) if data else None,
            output_dir=Path(merged.get("BTCGINI_OUTPUT_DIR") or "outputs"),
            processed_dir=Path(merged.get("BTCGINI_PROCESSED_DIR") or "data/processed"),
            seed=_int(merged, "BTCGINI_SEED", 42, 0),
            horizons=parse_horizons(horizons) if horizons else (1, 7),
            window=_int(merged, "BTCGINI_WINDOW", 30, 5),
            min_train=_int(merged, "BTCGINI_MIN_TRAIN", 730, 50),
            refit_every=_int(merged, "BTCGINI_REFIT_EVERY", 30, 1),
            cost_bps=_float(merged, "BTCGINI_COST_BPS", 10.0, 0.0),
        )
