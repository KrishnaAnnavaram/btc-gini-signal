# Data

Git does not store price data in this repository. Put files in `data/raw/` (git ignores it).
The code only reads raw files. Derived tables go to `data/processed/` (also ignored).

## Source 1: Coinbase Exchange API (recommended)

| Item | Value |
|---|---|
| Name | Coinbase Exchange public market data, product `BTC-USD`, daily candles |
| URL | https://api.exchange.coinbase.com/products/BTC-USD/candles (documentation: https://docs.cdp.coinbase.com/exchange/) |
| Terms | Coinbase API and market data terms. Do not redistribute the downloaded file |
| Key | Not needed |
| Day boundary | 00:00 UTC |
| Volume unit | BTC |
| Coverage | From 2015-07 (the start of the BTC-USD product) |

```bash
btc-gini-signal fetch --start 2016-01-01 --end 2025-12-31 --out data/raw/btc_usd_coinbase.csv
```

The command asks for 300 candles per request and refuses to overwrite an existing file.

## Source 2: Investing.com export

| Item | Value |
|---|---|
| Name | "Bitcoin Historical Data" export from Investing.com |
| URL | https://www.investing.com/crypto/bitcoin/historical-data |
| Terms | The site terms restrict redistribution of exported data. Keep the file local |
| Exchange | Not stated by the export (an index price). Record this in your notes |

The loader reads this layout without changes: `Date,Price,Open,High,Low,Vol.,Change %`, US month/day/year
dates, thousands separators, `K`/`M`/`B` volume suffixes and newest-first rows. `Price` is the close.
The loader does not use `Change %`.

## Generic layout

Any CSV with the columns `date,open,high,low,close` (and optional `volume`) works. Dates are ISO
(`2024-01-31`) or timestamps, one row per day.

## Validation rules

| Rule | Result |
|---|---|
| Duplicate dates, unsorted dates, missing or non-positive prices | Error |
| `low` above `open`/`close`, or `high` below them | Error |
| Calendar gaps | Warning (the return spans the gap) |
| No volume | Warning (no feature uses volume) |

## Offline data

The tests and the demo use the synthetic generator. It needs no download:

```bash
btc-gini-signal synth --days 2500 --seed 42 --out data/raw/synthetic_btc.csv
```
