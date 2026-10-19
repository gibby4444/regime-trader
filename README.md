# regime-trader

Volatility-regime allocation trader for US equities on Alpaca. A Gaussian HMM
detects market regimes; each regime maps to a target allocation/leverage; a risk
manager enforces sizing, exposure, and drawdown limits before orders are sent.

> Status: **skeleton only**. All modules contain typed stubs and docstrings;
> logic raises `NotImplementedError`.

## Layout

| Path | Purpose |
| --- | --- |
| `config/` | `settings.yaml` (all parameters), `credentials.yaml.example` |
| `core/` | HMM engine, regime strategies, risk manager, signal generator |
| `broker/` | Alpaca client, order executor, position tracker |
| `data/` | Market data (historical + streaming), feature engineering |
| `monitoring/` | Structured logging, `rich` dashboard, email/webhook alerts |
| `backtest/` | Walk-forward backtester, performance metrics, stress tests |
| `tests/` | Pytest stubs (currently skipped) incl. look-ahead bias checks |
| `main.py` | CLI entry point (`live` / `backtest` / `stress`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest            # dev
cp .env.example .env          # fill in Alpaca keys
cp config/credentials.yaml.example config/credentials.yaml  # optional: alert creds
```

## Usage (once implemented)

```bash
python main.py backtest
python main.py stress
python main.py live            # paper trading by default
pytest
```

## Design notes

- **No look-ahead:** features use trailing windows only; HMM uses forward-filtered
  (not smoothed) probabilities; backtest fills at the next bar.
- **Regime hygiene:** a regime must persist `stability_bars`; frequent switching
  (`flicker_threshold` within `flicker_window`) or low confidence scales size by
  `uncertainty_size_mult`.
- **Risk layering:** per-trade risk, per-position cap, gross exposure/leverage cap,
  daily/weekly reduce & halt thresholds, and a peak-drawdown kill switch.
