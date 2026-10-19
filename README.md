# regime-trader

Volatility-regime allocation trader for US equities on Alpaca. A Gaussian HMM
detects market regimes; each regime maps to a target allocation/leverage; a risk
manager enforces sizing, exposure, and drawdown limits before orders are sent.

> Status: HMM engine, strategies, features, risk drawdown states, and the
> walk-forward backtester (performance, benchmarks, stress tests) are implemented.
> Live trading, order execution and position tracking are still stubs.

## Layout

| Path | Purpose |
| --- | --- |
| `config/` | `settings.yaml` (all parameters), `credentials.yaml.example` |
| `core/` | HMM engine, regime strategies, risk manager, signal generator |
| `broker/` | Alpaca client, order executor, position tracker |
| `data/` | Market data (historical + streaming), feature engineering |
| `monitoring/` | Structured logging, `rich` dashboard, email/webhook alerts |
| `backtest/` | Walk-forward backtester, performance metrics, stress tests |
| `tests/` | Pytest suite (synthetic data) incl. look-ahead bias checks |
| `main.py` | CLI entry point (`live` / `backtest` / `stress`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest            # dev
cp .env.example .env          # fill in Alpaca keys
cp config/credentials.yaml.example config/credentials.yaml  # optional: alert creds
```

## Usage

```bash
# Walk-forward backtest (IS 252 / OOS 126 / step 126) on Alpaca history (needs ALPACA_* keys)
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare      # + benchmarks
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --stress-test  # + stress MC

# Offline: CSV file/dir (date, open, high, low, close, volume[, symbol]) or synthetic bars
python main.py backtest --csv data/SPY.csv --symbols SPY --start 2019-01-01 --compare
python main.py backtest --synthetic --compare --stress-test --n-sims 20

python main.py live            # not implemented yet
pytest
```

Backtest CSVs (equity curve, trade log, fills, regime history, summary tables,
stress results) are written to `--output-dir` (default `results/`, git-ignored).
History before `--start` is fetched for feature warm-up only.

## Design notes

- **No look-ahead:** features use trailing windows only; HMM uses forward-filtered
  (not smoothed) probabilities; backtest fills at the next bar.
- **Regime hygiene:** a regime must persist `stability_bars`; frequent switching
  (`flicker_threshold` within `flicker_window`) or low confidence scales size by
  `uncertainty_size_mult`.
- **Risk layering:** per-trade risk, per-position cap, gross exposure/leverage cap,
  daily/weekly reduce & halt thresholds, and a peak-drawdown kill switch.
