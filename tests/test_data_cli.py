import json

import numpy as np
import pandas as pd
import pytest

from btc_gini_signal import synthetic
from btc_gini_signal.cli import main
from btc_gini_signal.config import ConfigError, Settings, parse_horizons
from btc_gini_signal.data import (CSVSource, CoinbaseSource, DataError, SyntheticSource, load, parse_frame,
                                  parse_number, safe_output_path, validate)
from btc_gini_signal.forecast import direct_forecast
from btc_gini_signal.study import run_study

INVESTING = """Date,Price,Open,High,Low,Vol.,Change %
01/03/2024,"42,800.10","42,000.00","43,100.00","41,900.00",62.72K,1.90%
01/02/2024,"42,000.00","41,500.00","42,300.00","41,200.00",1.2M,1.20%
01/01/2024,"41,500.00","41,000.00","41,800.00","40,900.00",-,0.50%
"""


def test_parse_number_formats():
    assert parse_number("63,156.50") == 63156.5
    assert parse_number("62.72K") == pytest.approx(62720)
    assert parse_number("1.2M") == pytest.approx(1.2e6)
    assert parse_number("-1.71%") == pytest.approx(-1.71)
    assert np.isnan(parse_number("-"))
    with pytest.raises(DataError):
        parse_number("abc")


def test_investing_export_is_parsed_and_sorted(tmp_path):
    # Problem 9: the export format (separators, K/M volume, newest-first rows) is parsed and documented.
    path = tmp_path / "Bitcoin.csv"
    path.write_text(INVESTING, encoding="utf-8")
    before = path.read_bytes()
    data = load(CSVSource(path))
    assert list(data.frame["close"]) == [41500.0, 42000.0, 42800.1]
    assert data.frame["date"].is_monotonic_increasing
    assert data.frame["volume"].iloc[1] == pytest.approx(1.2e6)
    assert any("Investing.com" in n for n in data.provenance.notes)
    assert path.read_bytes() == before  # Problem 4: the raw file is never rewritten


def test_validation_errors():
    df = synthetic.generate(60, seed=1)
    df.loc[5, "low"] = df.loc[5, "high"] * 2
    df.loc[6, "close"] = -1
    errors, _ = validate(df)
    assert any("low <=" in e for e in errors) and any("> 0" in e for e in errors)
    dup = pd.concat([df.iloc[:3], df.iloc[:3]])
    assert any("duplicate" in e for e in validate(dup)[0])


def test_calendar_gaps_are_warnings():
    df = synthetic.generate(60, seed=2).drop(index=[10, 11])
    errors, warnings = validate(df.reset_index(drop=True))
    assert not errors and any("gap" in w for w in warnings)


def test_unknown_layout():
    with pytest.raises(DataError):
        parse_frame(pd.DataFrame({"a": [1], "b": [2]}))


def test_safe_output_path_refuses_overwrite(tmp_path):
    raw = tmp_path / "Bitcoin.csv"
    raw.write_text("x")
    with pytest.raises(DataError):
        safe_output_path(raw, [raw])
    with pytest.raises(DataError):
        safe_output_path(tmp_path / "raw" / "f.csv", [])
    assert safe_output_path(tmp_path / "processed" / "f.csv", [raw]).name == "f.csv"


def test_coinbase_adapter_with_fake_http():
    calls = []

    def fake_get(url):
        calls.append(url)
        day = 1704067200  # 2024-01-01 UTC
        return json.dumps([[day + 86400 * i, 99.0, 101.0, 100.0, 100.5, 7.0] for i in range(3)][::-1])

    data = load(CoinbaseSource("2024-01-01", "2024-01-03", http_get=fake_get))
    assert len(data.frame) == 3 and calls[0].startswith("https://api.exchange.coinbase.com/products/BTC-USD")
    assert data.frame["close"].iloc[0] == 100.5 and data.provenance.source == "coinbase"


def test_coinbase_adapter_rejects_error_payload():
    with pytest.raises(DataError):
        CoinbaseSource("2024-01-01", "2024-01-03", http_get=lambda u: '{"message": "rate limited"}').read()


def test_settings_and_horizons():
    s = Settings.from_env({"BTCGINI_HORIZONS": "7,1", "BTCGINI_WINDOW": "20", "BTCGINI_COST_BPS": "5"})
    assert s.horizons == (1, 7) and s.window == 20 and s.cost_bps == 5
    with pytest.raises(ConfigError):
        parse_horizons("0,1")  # a same-day target is not a forecast
    with pytest.raises(ConfigError):
        Settings.from_env({"BTCGINI_WINDOW": "2"})


def test_synthetic_generator_is_deterministic():
    assert synthetic.generate(100, seed=3).equals(synthetic.generate(100, seed=3))
    assert validate(synthetic.generate(100, seed=3))[0] == []


def test_study_detects_a_planted_inequality_signal():
    data = load(SyntheticSource(n_days=2000, seed=11, signal=6.0))
    settings = Settings(horizons=(1,), min_train=500, refit_every=60)
    study = run_study(data, settings, models=("random_walk", "ridge"))
    ridge = next(a for a in study.ablation if a["model"] == "ridge")
    assert ridge["dm_statistic"] > 2 and ridge["p_value"] < 0.05


def test_study_without_signal_does_not_beat_the_random_walk():
    data = load(SyntheticSource(n_days=1200, seed=12, signal=0.0))
    settings = Settings(horizons=(1,), min_train=500, refit_every=100)
    study = run_study(data, settings, models=("random_walk", "ridge"))
    for r in study.rows:
        assert r["dm_vs_random_walk"]["p_value"] > 0.01 or r["model"] == "random_walk"


def test_direct_forecast_gives_one_row_per_horizon():
    # Problem 5: a real multi-step forecast, with no manual reruns.
    data = load(SyntheticSource(n_days=400, seed=13))
    table = direct_forecast(data, [1, 3, 7], model="ridge")
    assert list(table["horizon_days"]) == [1, 3, 7]
    last = data.frame["close"].iloc[-1]
    assert np.allclose(table["predicted_close"], last * np.exp(table["predicted_log_return"]))
    assert table["target_date"].iloc[2] > table["origin_date"].iloc[0]


def test_cli_end_to_end(tmp_path, capsys):
    raw = tmp_path / "raw" / "s.csv"
    assert main(["synth", "--days", "700", "--out", str(raw)]) == 0
    assert main(["validate", "--data", str(raw)]) == 0
    feats = tmp_path / "processed" / "features.csv"
    assert main(["features", "--data", str(raw), "--out", str(feats), "--horizons", "1,7"]) == 0
    table = pd.read_csv(feats)
    assert {"gini_abs_ret", "kolkata_abs_ret", "y_1d", "y_7d"} <= set(table.columns)
    out = tmp_path / "rep"
    assert main(["evaluate", "--data", str(raw), "--horizons", "1", "--min-train", "300", "--refit-every", "100",
                 "--models", "random_walk,ridge", "--out", str(out)]) == 0
    assert "Not financial advice" in (out / "report.md").read_text(encoding="utf-8")
    assert main(["forecast", "--data", str(raw), "--horizons", "1,7"]) == 0


def test_cli_refuses_to_overwrite_raw_input(tmp_path, capsys):
    raw = tmp_path / "s.csv"
    main(["synth", "--days", "100", "--out", str(raw)])
    assert main(["features", "--data", str(raw), "--out", str(raw)]) == 2
    assert main(["validate", "--data", str(tmp_path / "missing.csv")]) == 2
