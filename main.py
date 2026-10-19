"""regime-trader entry point.

Modes:
    live      Run the scheduled trading loop against Alpaca (paper by default).
    backtest  Run the walk-forward backtest and print a performance report.
    stress    Run stress-test scenarios.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import schedule  # noqa: F401
import yaml  # noqa: F401
from dotenv import load_dotenv  # noqa: F401

DEFAULT_CONFIG = Path(__file__).parent / "config" / "settings.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load settings.yaml and overlay credentials from .env / credentials.yaml."""
    raise NotImplementedError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (mode, config path, log level, --no-dashboard)."""
    raise NotImplementedError


def run_live(config: dict[str, Any], dashboard: bool = True) -> None:
    """Build components, then each bar: fetch data -> features -> signals -> risk -> orders."""
    raise NotImplementedError


def run_backtest(config: dict[str, Any]) -> None:
    """Fetch history, run the walk-forward backtester, and print the report."""
    raise NotImplementedError


def run_stress(config: dict[str, Any]) -> None:
    """Run all default stress scenarios and summarise drawdowns."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
