"""Daily OHLCV sources behind one interface, parsing of known export formats, and validation.

The canonical frame has the columns `date` (UTC calendar day), `open`, `high`, `low`, `close`
and `volume` (can be NaN), sorted by date with one row per day. Raw files are only read,
never written.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from . import synthetic

CANONICAL = ["date", "open", "high", "low", "close", "volume"]


class DataError(ValueError):
    """The price table breaks the contract."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Provenance:
    source: str
    detail: str
    day_boundary: str = "UTC"
    notes: list[str] = field(default_factory=list)


class PriceSource(Protocol):
    name: str

    def read(self) -> tuple[pd.DataFrame, Provenance]: ...


_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}


def parse_number(text) -> float:
    """Parse '63,156.50', '62.72K', '1.2M', '-1.71%' or '-' into a float (percent stays in percent units)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return float("nan")
    if isinstance(text, (int, float, np.integer, np.floating)):
        return float(text)
    s = str(text).strip().replace(",", "")
    if s in {"", "-", "nan", "NaN"}:
        return float("nan")
    s = s.rstrip("%")
    mult = 1.0
    if s and s[-1].upper() in _SUFFIX:
        mult = _SUFFIX[s[-1].upper()]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError as exc:
        raise DataError([f"cannot parse number {text!r}"]) from exc


def _detect_format(columns: list[str]) -> str:
    cols = {c.strip().lower() for c in columns}
    if {"date", "price", "open", "high", "low"} <= cols:
        return "investing"
    if {"date", "open", "high", "low", "close"} <= cols:
        return "generic"
    raise DataError([f"unknown CSV layout with columns {columns}"])


def parse_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert an Investing.com export or a generic OHLCV table to the canonical frame."""
    notes: list[str] = []
    raw = raw.rename(columns=lambda c: c.strip())
    fmt = _detect_format(list(raw.columns))
    lower = {c.lower(): c for c in raw.columns}
    if fmt == "investing":
        notes.append("Investing.com layout: 'Price' is the close, US month/day/year dates, K/M/B volume suffixes")
        dates = pd.to_datetime(raw[lower["date"]], format="mixed", dayfirst=False)
        close_col = lower["price"]
        vol_col = lower.get("vol.") or lower.get("volume")
    else:
        dates = pd.to_datetime(raw[lower["date"]], utc=True, format="mixed").dt.tz_localize(None)
        close_col = lower["close"]
        vol_col = lower.get("volume")
    out = pd.DataFrame({
        "date": dates.dt.normalize(),
        "open": raw[lower["open"]].map(parse_number),
        "high": raw[lower["high"]].map(parse_number),
        "low": raw[lower["low"]].map(parse_number),
        "close": raw[close_col].map(parse_number),
        "volume": raw[vol_col].map(parse_number) if vol_col else np.nan,
    })
    if out["date"].is_monotonic_decreasing and len(out) > 1:
        notes.append("rows were newest-first and are now sorted oldest-first")
    out = out.sort_values("date").reset_index(drop=True)
    return out, notes


def validate(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a canonical frame."""
    errors: list[str] = []
    warnings: list[str] = []
    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"], warnings
    if len(df) < 2:
        errors.append("need at least 2 rows")
        return errors, warnings
    if df["date"].duplicated().any():
        errors.append(f"{int(df['date'].duplicated().sum())} duplicate dates")
    if not df["date"].is_monotonic_increasing:
        errors.append("dates are not sorted")
    prices = df[["open", "high", "low", "close"]]
    if prices.isna().any().any():
        errors.append(f"{int(prices.isna().sum().sum())} missing prices")
    if (prices <= 0).any().any():
        errors.append("prices must be > 0")
    bad = (df["low"] > df[["open", "close"]].min(axis=1) * (1 + 1e-9)) | \
          (df["high"] < df[["open", "close"]].max(axis=1) * (1 - 1e-9))
    if bad.any():
        errors.append(f"{int(bad.sum())} rows break low <= open, close <= high")
    gaps = df["date"].diff().dt.days.iloc[1:]
    if (gaps > 1).any():
        warnings.append(f"{int((gaps > 1).sum())} calendar gaps (largest {int(gaps.max())} days); "
                        "returns span the gap")
    if df["volume"].isna().all():
        warnings.append("no volume data")
    return errors, warnings


@dataclass
class PriceData:
    frame: pd.DataFrame
    provenance: Provenance
    warnings: list[str]

    @property
    def sha256(self) -> str:
        hashed = pd.util.hash_pandas_object(self.frame, index=False).to_numpy()
        return hashlib.sha256(hashed.tobytes()).hexdigest()


@dataclass
class CSVSource:
    path: Path
    name: str = "csv"

    def read(self) -> tuple[pd.DataFrame, Provenance]:
        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. See data/README.md, or use `--data synthetic`.")
        raw = pd.read_csv(path, dtype=str)
        frame, notes = parse_frame(raw)
        return frame, Provenance("csv", str(path), notes=notes)


@dataclass
class SyntheticSource:
    n_days: int = 2500
    seed: int = 42
    signal: float = 0.0
    name: str = "synthetic"

    def read(self) -> tuple[pd.DataFrame, Provenance]:
        frame = synthetic.generate(self.n_days, self.seed, signal=self.signal)
        return frame, Provenance("synthetic", f"n_days={self.n_days} seed={self.seed} signal={self.signal}",
                                 notes=["SYNTHETIC prices from a GARCH-style generator with jumps"])


HttpGet = Callable[[str], str]


def _urllib_get(url: str) -> str:  # pragma: no cover - network
    req = urllib.request.Request(url, headers={"User-Agent": "btc-gini-signal/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


@dataclass
class CoinbaseSource:
    """Daily candles from the public Coinbase Exchange API (no key). 300 candles per request."""

    start: str
    end: str
    product: str = "BTC-USD"
    http_get: HttpGet = _urllib_get
    base_url: str = "https://api.exchange.coinbase.com"
    name: str = "coinbase"

    def read(self) -> tuple[pd.DataFrame, Provenance]:
        start = datetime.fromisoformat(self.start).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(self.end).replace(tzinfo=timezone.utc)
        if end <= start:
            raise DataError(["end must be after start"])
        rows: list[list[float]] = []
        cursor = start
        while cursor < end:
            stop = min(cursor + timedelta(days=299), end)
            url = (f"{self.base_url}/products/{self.product}/candles?granularity=86400"
                   f"&start={cursor.strftime('%Y-%m-%dT%H:%M:%SZ')}&end={stop.strftime('%Y-%m-%dT%H:%M:%SZ')}")
            payload = json.loads(self.http_get(url))
            if not isinstance(payload, list):
                raise DataError([f"unexpected API answer: {str(payload)[:200]}"])
            rows.extend(payload)
            cursor = stop + timedelta(days=1)
        if not rows:
            raise DataError(["the API returned no candles"])
        arr = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        frame = pd.DataFrame({
            "date": pd.to_datetime(arr["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize(),
            "open": arr["open"].astype(float), "high": arr["high"].astype(float),
            "low": arr["low"].astype(float), "close": arr["close"].astype(float),
            "volume": arr["volume"].astype(float),
        }).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        return frame, Provenance("coinbase", f"{self.base_url} {self.product} daily candles",
                                 notes=["exchange: Coinbase Exchange", "volume in BTC"])


def load(source: PriceSource) -> PriceData:
    frame, prov = source.read()
    errors, warnings = validate(frame)
    if errors:
        raise DataError(errors)
    return PriceData(frame[CANONICAL].copy(), prov, warnings)


def source_for(path: str | Path | None, n_days: int = 2500, seed: int = 42, signal: float = 0.0) -> PriceSource:
    if path is None or str(path) == "synthetic":
        return SyntheticSource(n_days, seed, signal)
    return CSVSource(Path(path))


def safe_output_path(out: Path, inputs: list[Path]) -> Path:
    """Refuse to write over an input file (raw data is read-only)."""
    out_r = Path(out).resolve()
    for p in inputs:
        if p is not None and Path(p).resolve() == out_r:
            raise DataError([f"refusing to overwrite the input file {p}"])
    if re.search(r"(^|[\\/])raw([\\/]|$)", str(out_r.parent)):
        raise DataError(["refusing to write into a raw data folder"])
    return out_r
