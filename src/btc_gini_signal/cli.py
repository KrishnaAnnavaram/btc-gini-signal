"""Command line interface: `btc-gini-signal <command>`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from . import DISCLAIMER, __version__, report
from .config import ConfigError, Settings, parse_horizons
from .data import CSVSource, CoinbaseSource, DataError, load, safe_output_path, source_for
from .features import build_features, make_target
from .forecast import direct_forecast
from .models import MODELS
from .study import DEFAULT_MODELS, run_study


def _settings(args) -> Settings:
    return Settings.from_env().with_overrides(
        seed=getattr(args, "seed", None),
        horizons=parse_horizons(args.horizons) if getattr(args, "horizons", None) else None,
        window=getattr(args, "window", None),
        min_train=getattr(args, "min_train", None),
        refit_every=getattr(args, "refit_every", None),
        cost_bps=getattr(args, "cost_bps", None),
    )


def _data(args, settings: Settings):
    path = args.data or (str(settings.data_path) if settings.data_path else None)
    if path is None or path == "synthetic":
        print(f"[btc-gini-signal] using {args.synthetic_days} SYNTHETIC days (signal {args.signal}).",
              file=sys.stderr)
    data = load(source_for(path, args.synthetic_days, settings.seed, args.signal))
    for w in data.warnings:
        print(f"[btc-gini-signal] warning: {w}", file=sys.stderr)
    return data


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", help="CSV path or 'synthetic' (default: BTCGINI_DATA, else synthetic)")
    p.add_argument("--synthetic-days", type=int, default=2500)
    p.add_argument("--signal", type=float, default=0.0, help="planted inequality signal for synthetic data")
    p.add_argument("--seed", type=int)
    p.add_argument("--window", type=int, help="inequality window in days")


def cmd_synth(args) -> int:
    from .synthetic import generate
    df = generate(args.days, args.seed, signal=args.signal)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"wrote {len(df)} synthetic days to {out}")
    return 0


def cmd_fetch(args) -> int:
    frame, prov = CoinbaseSource(args.start, args.end, args.product).read()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise DataError([f"{out} exists; raw files are not overwritten (use a new name or --force)"])
    frame.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"wrote {len(frame)} days from {prov.detail} to {out}")
    return 0


def cmd_validate(args) -> int:
    try:
        data = load(CSVSource(Path(args.data)))
    except DataError as exc:
        for e in exc.errors:
            print(f"error: {e}")
        print("data: FAILED")
        return 1
    f = data.frame
    print(f"rows: {len(f)}  {f['date'].iloc[0].date()} .. {f['date'].iloc[-1].date()}")
    for n in data.provenance.notes:
        print(f"note: {n}")
    for w in data.warnings:
        print(f"warning: {w}")
    print("data: OK")
    return 0


def cmd_features(args) -> int:
    settings = _settings(args)
    data = _data(args, settings)
    out = Path(args.out) if args.out else settings.processed_dir / "features.csv"
    out = safe_output_path(out, [Path(args.data)] if args.data and args.data != "synthetic" else [])
    table = build_features(data.frame, settings.window)
    for h in settings.horizons:
        table[f"y_{h}d"] = make_target(data.frame, h).to_numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, date_format="%Y-%m-%d")
    print(f"wrote {len(table)} rows x {table.shape[1]} columns to {out}")
    return 0


def cmd_evaluate(args) -> int:
    settings = _settings(args)
    data = _data(args, settings)
    models = [m.strip() for m in args.models.split(",")] if args.models else list(DEFAULT_MODELS)
    for m in models:
        if m not in MODELS:
            raise ConfigError(f"unknown model {m!r}; choose from {sorted(MODELS)}")
    study = run_study(data, settings, models)
    print(f"{'h':<3}{'model':<12}{'features':<17}{'RMSE':<9}{'R2 vs RW':<10}{'DM':<7}{'p':<6}")
    for r in study.rows:
        dm = r["dm_vs_random_walk"]
        print(f"{r['horizon']:<3}{r['model']:<12}{r['feature_set']:<17}{r['rmse']:<9.5f}{r['r2_oos']:<10.4f}"
              f"{dm['statistic']:<7.2f}{dm['p_value']:<6.3f}")
    print("ablation (positive DM = inequality features reduce the squared error):")
    for a in study.ablation:
        print(f"  h={a['horizon']} {a['model']:<6} DM={a['dm_statistic']:+.2f} p={a['p_value']:.3f}")
    out = Path(args.out) if args.out else settings.output_dir
    paths = report.write(study, data, settings, out)
    print(f"wrote {paths['report']}")
    print(DISCLAIMER)
    return 0


def cmd_forecast(args) -> int:
    settings = _settings(args)
    data = _data(args, settings)
    table = direct_forecast(data, list(settings.horizons), args.model, args.feature_set, settings.window,
                            settings.seed)
    print(table.to_string(index=False))
    print(DISCLAIMER)
    return 0


def cmd_demo(args) -> int:
    args.data = "synthetic"
    args.models = None
    return cmd_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btc-gini-signal",
                                     description="Walk-forward test of Gini and Kolkata features for Bitcoin returns.")
    parser.add_argument("--version", action="version", version=f"btc-gini-signal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write a synthetic OHLCV CSV")
    p.add_argument("--days", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--signal", type=float, default=0.0)
    p.add_argument("--out", default="data/raw/synthetic_btc.csv")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("fetch", help="download daily candles from the public Coinbase Exchange API")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", required=True, help="end date, for example 2024-12-31")
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--out", default="data/raw/btc_usd_coinbase.csv")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("validate", help="check a price CSV")
    p.add_argument("--data", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("features", help="write the feature and target table to data/processed")
    _common(p)
    p.add_argument("--horizons")
    p.add_argument("--out")
    p.set_defaults(func=cmd_features)

    for name, func, text in (("evaluate", cmd_evaluate, "walk-forward ablation study"),
                             ("demo", cmd_demo, "offline demo on synthetic prices")):
        p = sub.add_parser(name, help=text)
        _common(p)
        p.add_argument("--horizons", help="for example 1,7")
        p.add_argument("--min-train", type=int)
        p.add_argument("--refit-every", type=int)
        p.add_argument("--cost-bps", type=float)
        p.add_argument("--models", help=f"comma list from {sorted(MODELS)}")
        p.add_argument("--out")
        p.set_defaults(func=func)

    p = sub.add_parser("forecast", help="direct multi-step forecast from the last day")
    _common(p)
    p.add_argument("--horizons", default="1,7")
    p.add_argument("--model", default="ridge", choices=sorted(MODELS))
    p.add_argument("--feature-set", default="base+inequality", choices=["base", "base+inequality", "inequality"])
    p.set_defaults(func=cmd_forecast)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ConfigError, DataError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
