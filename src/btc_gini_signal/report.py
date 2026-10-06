"""Write the study as JSON and Markdown."""

from __future__ import annotations

import json
from pathlib import Path

from . import DISCLAIMER, __version__
from .config import Settings
from .data import PriceData
from .study import StudyResult


def _f(v, digits=4):
    if v is None or v != v:  # NaN
        return "-"
    return f"{v:.{digits}f}"


def render(study: StudyResult, data: PriceData, settings: Settings) -> str:
    frame = data.frame
    lines = [
        "# btc-gini-signal walk-forward report", "",
        f"> {DISCLAIMER}", "",
        f"- Data: `{data.provenance.source}` ({data.provenance.detail}), {len(frame)} days, "
        f"{frame['date'].iloc[0].date()} to {frame['date'].iloc[-1].date()}, sha256 `{data.sha256[:12]}`",
        *[f"- Note: {n}" for n in data.provenance.notes],
        f"- Walk-forward: expanding window, first {settings.min_train} usable rows for the first fit, "
        f"refit every {settings.refit_every} days, embargo = horizon",
        f"- Inequality window: {settings.window} days of absolute log returns. Cost: {settings.cost_bps} bps per unit turnover",
        f"- btc-gini-signal {__version__}", "",
        "## Forecast accuracy (log returns, out-of-sample)", "",
        "| h | Model | Features | n | MAE | RMSE | R2 vs RW | Dir. acc. | DM vs RW | p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in study.rows:
        dm = r["dm_vs_random_walk"]
        lines.append(f"| {r['horizon']} | {r['model']} | {r['feature_set']} | {r['n']} | {_f(r['mae'])} | "
                     f"{_f(r['rmse'])} | {_f(r['r2_oos'])} | {_f(r['directional_accuracy'], 3)} | "
                     f"{_f(dm['statistic'], 2)} | {_f(dm['p_value'], 3)} |")
    lines += ["", "## Ablation: base vs base+inequality (same rows)", "",
              "| h | Model | n | RMSE base | RMSE + inequality | DM (positive = inequality helps) | p |",
              "|---|---|---|---|---|---|---|"]
    for a in study.ablation:
        lines.append(f"| {a['horizon']} | {a['model']} | {a['n']} | {_f(a['rmse_base'], 5)} | "
                     f"{_f(a['rmse_with_inequality'], 5)} | {_f(a['dm_statistic'], 2)} | {_f(a['p_value'], 3)} |")
    if study.backtests:
        lines += ["", "## Sign-strategy backtest (h = 1, after costs)", "",
                  "| Model | Features | Total log return | Sharpe | Max drawdown | Turnover | Buy and hold log return | Buy and hold Sharpe |",
                  "|---|---|---|---|---|---|---|---|"]
        for b in study.backtests:
            lines.append(f"| {b['model']} | {b['feature_set']} | {_f(b['total_log_return'], 3)} | {_f(b['sharpe'], 2)} | "
                         f"{_f(b['max_drawdown'], 3)} | {_f(b['turnover'], 3)} | {_f(b['buy_hold_log_return'], 3)} | "
                         f"{_f(b['buy_hold_sharpe'], 2)} |")
    lines += ["", "## How to read this report", "",
              "- R2 vs RW below 0 means the model is worse than predicting no change.",
              "- The DM tests are not adjusted for the number of comparisons.",
              "- The ablation answers the research question: do Gini and Kolkata features reduce the error?", ""]
    return "\n".join(lines)


def write(study: StudyResult, data: PriceData, settings: Settings, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"report": out_dir / "report.md", "results": out_dir / "results.json"}
    paths["report"].write_text(render(study, data, settings), encoding="utf-8")
    record = {"version": __version__, "disclaimer": DISCLAIMER, "data_sha256": data.sha256,
              "provenance": data.provenance.__dict__, "settings": {k: str(v) for k, v in settings.__dict__.items()},
              "rows": study.rows, "ablation": study.ablation, "backtests": study.backtests}
    paths["results"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    return paths
