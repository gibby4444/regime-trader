"""regime-trader entry point.

Modes:
    live      Run the scheduled trading loop against Alpaca (paper by default).
    backtest  Run the walk-forward backtest and print a performance report.
              --compare adds benchmarks; --stress-test adds stress scenarios.
    stress    Shorthand for ``backtest --stress-test``.

Data source for backtests, in priority order: --csv PATH, --synthetic, Alpaca
historical bars (requires ALPACA_API_KEY / ALPACA_SECRET_KEY).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG = Path(__file__).parent / "config" / "settings.yaml"
DEFAULT_CREDENTIALS = Path(__file__).parent / "config" / "credentials.yaml"
# Calendar days fetched before --start for feature warm-up (200-bar SMA + 252-bar z-score ~ 451 bars).
WARMUP_CALENDAR_DAYS = 700

logger = logging.getLogger("regime_trader")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load settings.yaml and overlay credentials from .env / credentials.yaml.

    Credentials land under ``config["credentials"]``; environment variables win
    over credentials.yaml.
    """
    load_dotenv()
    with Path(path).open() as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}
    creds: dict[str, Any] = {}
    if DEFAULT_CREDENTIALS.exists():
        with DEFAULT_CREDENTIALS.open() as fh:
            creds = yaml.safe_load(fh) or {}
    env = {
        "alpaca_api_key": os.environ.get("ALPACA_API_KEY"),
        "alpaca_secret_key": os.environ.get("ALPACA_SECRET_KEY"),
        "alpaca_paper": os.environ.get("ALPACA_PAPER"),
    }
    creds.update({k: v for k, v in env.items() if v})
    config["credentials"] = creds
    return config


def _add_backtest_args(p: argparse.ArgumentParser, stress_default: bool = False) -> None:
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Symbols to trade; the first drives the HMM (default: SPY)")
    p.add_argument("--start", default=None, help="First in-sample bar date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Last bar date (YYYY-MM-DD)")
    p.add_argument("--csv", default=None, help="Offline OHLCV CSV file or directory of <SYMBOL>.csv")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic regime-switching bars")
    p.add_argument("--synthetic-bars", type=int, default=2000)
    p.add_argument("--compare", action="store_true", help="Benchmarks: buy & hold, 200 SMA, random")
    p.add_argument("--stress-test", action="store_true", default=stress_default,
                   help="Crash, gap and regime-misclassification Monte Carlo")
    p.add_argument("--n-sims", type=int, default=100, help="Simulations per stress scenario / random seeds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="results", help="Directory for CSV outputs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (mode, config path, log level, backtest options)."""
    parser = argparse.ArgumentParser(prog="regime-trader", description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("live", help="Run the live/paper trading loop")
    live.add_argument("--no-dashboard", action="store_true")
    _add_backtest_args(sub.add_parser("backtest", help="Walk-forward backtest"))
    _add_backtest_args(sub.add_parser("stress", help="Backtest plus stress tests"), stress_default=True)
    return parser.parse_args(argv)


def load_bars(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Fetch bars from --csv, --synthetic, or Alpaca."""
    symbols = [s.upper() for s in (args.symbols or ["SPY"])]
    if args.csv:
        from data.market_data import load_bars_csv

        return load_bars_csv(args.csv, symbols)
    if args.synthetic:
        from data.synthetic import make_synthetic_bars

        start = (pd.Timestamp(args.start) - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)) if args.start else "2017-01-01"
        return {
            s: make_synthetic_bars(args.synthetic_bars, seed=args.seed + i, start=str(pd.Timestamp(start).date()))[0]
            for i, s in enumerate(symbols)
        }
    from broker.alpaca_client import AlpacaClient
    from data.market_data import MarketDataFeed

    if not AlpacaClient.has_env_credentials():
        raise SystemExit("No data source: set ALPACA_API_KEY/ALPACA_SECRET_KEY, or pass --csv PATH or --synthetic")
    start = pd.Timestamp(args.start or "2019-01-01") - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)
    end = pd.Timestamp(args.end) if args.end else None
    feed = MarketDataFeed(AlpacaClient.from_env(), symbols, config.get("broker", {}).get("timeframe", "1Day"),
                          cache_dir="data/cache")
    return feed.get_history(start, end)


def run_live(config: dict[str, Any], dashboard: bool = True) -> None:
    """Build components, then each bar: fetch data -> features -> signals -> risk -> orders."""
    raise NotImplementedError("live trading is not implemented yet")


def run_backtest(config: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Fetch history, run the walk-forward backtester, and print the report (+ benchmarks/stress)."""
    from rich.console import Console

    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import PerformanceReport, compare_benchmarks
    from backtest.stress_test import StressTester, summarize

    args = args or parse_args(["backtest"])
    console = Console()
    bars = load_bars(args, config)
    bt = WalkForwardBacktester.from_settings(config)
    console.print(f"[bold]Walk-forward backtest[/bold] {list(bars)} "
                  f"IS={bt.config.train_window} OOS={bt.config.test_window} step={bt.config.step_size}")
    result = bt.run(bars, start=args.start, end=args.end)
    for i, f in enumerate(result.folds):
        console.print(f"  window {i}: IS {f.train_start.date()}..{f.train_end.date()} "
                      f"OOS {f.test_start.date()}..{f.test_end.date()} n_states={f.n_states}")

    report = PerformanceReport.from_result(result, bt.config.risk_free_rate)
    out: dict[str, Any] = {"result": result, "report": report}
    if args.compare:
        report.benchmarks = compare_benchmarks(
            result, bars, bt.config, bt.risk_config, n_random=args.n_sims, seed=args.seed
        )
    report.print_summary(console)
    paths = report.to_csv(result, args.output_dir)
    out["paths"] = paths

    if args.stress_test:
        tester = StressTester(bt, seed=args.seed, n_sims=args.n_sims)
        outcomes = tester.run_all(bars, baseline=result, start=args.start, end=args.end)
        table = summarize(outcomes, result)
        from backtest.performance import _frame_table

        console.print(_frame_table("Stress Tests", table.rename_axis("Scenario"),
                                   {"mean_return", "p5_return", "mean_max_dd", "worst_max_dd",
                                    "worst_window_dd", "kill_rate"}))
        stress_path = Path(args.output_dir) / "stress_summary.csv"
        table.to_csv(stress_path, index_label="scenario")
        for name, o in outcomes.items():
            o.sims.to_csv(Path(args.output_dir) / f"stress_{name}_sims.csv", index=False)
        paths["stress_summary"] = stress_path
        out["stress"] = table
    console.print(f"CSV outputs written to {Path(args.output_dir).resolve()}")
    return out


def run_stress(config: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Run the backtest followed by all stress scenarios."""
    args = args or parse_args(["stress"])
    args.stress_test = True
    return run_backtest(config, args)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    if args.mode == "live":
        run_live(config, dashboard=not args.no_dashboard)
    elif args.mode == "backtest":
        run_backtest(config, args)
    else:
        run_stress(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
