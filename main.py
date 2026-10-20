"""Trading, training and offline research entry points; imports never connect."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

DEFAULT_CONFIG=Path(__file__).parent/'config'/'settings.yaml'

def load_config(path=DEFAULT_CONFIG):
    """Read only the supplied YAML settings; no credential or environment files."""
    with Path(path).open() as stream:
        return yaml.safe_load(stream)

def parse_args(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='live', choices=['live','train-only','backtest','stress'])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--backtest', action='store_true')
    modes.add_argument('--train-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--symbols',nargs='+')
    parser.add_argument('--start',default='2019-01-01')
    parser.add_argument('--end')
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--stress-test',action='store_true')
    parser.add_argument('--seeds',type=int)
    parser.add_argument('--simulations',type=int)
    parser.add_argument('--points',type=int)
    parser.add_argument('--output',type=Path,default=Path('backtest_results'))
    return parser.parse_args(argv)

def build_backtester(config):
    from backtest.backtester import BacktestConfig,WalkForwardBacktester
    from core.hmm_engine import HMMEngine,HMMConfig
    from core.regime_strategies import StrategyOrchestrator,StrategyConfig
    from core.risk_manager import RiskManager,RiskConfig
    from data.feature_engineering import FeatureEngineer,FeatureConfig
    # Regime mapping is populated immediately after the first IS fit.
    strategy=StrategyOrchestrator(StrategyConfig.from_dict(config.get('strategy'),config.get('hmm')))
    return WalkForwardBacktester(BacktestConfig.from_dict(config['backtest']),HMMEngine(HMMConfig.from_dict(config['hmm'])),
                                 strategy,RiskManager(RiskConfig.from_dict(config['risk'])),
                                 FeatureEngineer(FeatureConfig.from_dict(config.get('features'))))

def run_backtest(config,args=None):
    from data.market_data import load_yfinance_history
    from backtest.performance import PerformanceReport
    from backtest.stress_test import StressTester
    args=args or parse_args(['backtest'])
    bars=load_yfinance_history(args.symbols or config['broker']['symbols'],args.start,args.end)
    backtester=build_backtester(config)
    result=backtester.run(bars)
    report=PerformanceReport.from_result(result,backtester.config.risk_free_rate)
    report.print_summary()
    result.to_csv(args.output)
    report.regime_stats.to_csv(args.output/'regime_metrics.csv')
    report.confidence_stats.to_csv(args.output/'confidence_metrics.csv')
    if args.compare:
        benchmarks,random=backtester.compare(bars,args.seeds)
        for name,benchmark in benchmarks.items():
            print(name)
            PerformanceReport.from_result(benchmark).print_summary()
            benchmark.to_csv(args.output/name)
        print('Random allocations (same risk rules):')
        print(random.to_string())
        random.to_csv(args.output/'random_benchmark.csv')
    if args.stress_test or args.mode=='stress':
        stress=StressTester(backtester,seed=42).run_all(bars,args.simulations,args.points)
        print(stress.groupby('scenario')[['total_return','max_drawdown','breaker_triggers']].agg(['mean','std']).to_string())
        stress.to_csv(args.output/'stress_results.csv',index=False)
    return result

def run_stress(config):
    return run_backtest(config,parse_args(['stress']))

def build_live_loop(config, *, dry_run=False):
    """Construct lazy adapters; connect only on explicit execution."""
    from datetime import datetime, timedelta, timezone
    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from data.market_data import MarketDataFeed
    from data.feature_engineering import FeatureEngineer, FeatureConfig
    from core.hmm_engine import HMMEngine, HMMConfig
    from core.regime_strategies import StrategyOrchestrator, StrategyConfig
    from core.risk_manager import RiskManager, RiskConfig, PortfolioState
    from core.trading_loop import TradingLoop, retry_api
    from monitoring.status_store import StatusStore
    from monitoring.dashboard import Dashboard
    from monitoring.alerts import AlertManager
    import time

    from monitoring.logger import setup_logging
    monitoring = config.get('monitoring', {})
    setup_logging(log_dir=monitoring.get('log_dir', 'logs'))
    webhook = monitoring.get('webhook', {})
    client = AlpacaClient(config['broker'])
    symbols = config['broker']['symbols']
    feed = MarketDataFeed(client, symbols, config['broker'].get('live_timeframe', '5Min'))
    features = FeatureEngineer(FeatureConfig.from_dict(config.get('features')))
    engine = HMMEngine(HMMConfig.from_dict(config.get('hmm')))
    risk = RiskManager(RiskConfig.from_dict(config['risk']))
    portfolio = PortfolioState(0, 0, 0)
    tracker = PositionTracker(client, risk, portfolio)
    strategy = StrategyOrchestrator(StrategyConfig.from_dict(config.get('strategy'), config.get('hmm')))

    def train(model):
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        daily = MarketDataFeed(client, symbols, '1Day')
        bars = retry_api(lambda: daily.get_history(end-timedelta(days=2000), end), time.sleep)
        source = bars[symbols[0]]
        matrix = features.build_features(source)
        model.fit(matrix, features.returns(source).reindex(matrix.index))

    def warm_history():
        end = datetime.now(timezone.utc)
        history = feed.get_history(end-timedelta(days=30), end)
        # REST timestamps denote interval starts; omit the currently forming interval.
        import pandas as pd
        cutoff = pd.Timestamp(end).floor(config['broker'].get('live_timeframe', '5Min').replace('Min', 'min'))
        return {s: b.loc[b.index < cutoff] for s, b in history.items()}

    return TradingLoop(config, client=client, feed=feed, engine=engine, features=features,
                       strategy=strategy, risk=risk, tracker=tracker,
                       executor=OrderExecutor(client, risk, portfolio), dashboard=Dashboard(monitoring.get('dashboard_refresh_seconds', 5)),
                       alerts=AlertManager(monitoring.get('alert_rate_limit_minutes', 15),
                                           monitoring.get('email'),
                                           webhook.get('url') if webhook.get('enabled', False) else None), train=train, dry_run=dry_run, warm_history=warm_history, status_store=StatusStore())


def main(argv=None):
    args=parse_args(argv)
    config = load_config(args.config)
    import logging
    logging.basicConfig(level=logging.INFO)
    if args.train_only or args.mode == 'train-only':
        loop = build_live_loop(config, dry_run=True)
        loop.read(loop.client.connect)
        loop.train(loop.engine)
        loop.engine.save(loop.model_path)
    elif args.backtest or args.mode in ('backtest', 'stress') or args.stress_test or args.compare:
        run_backtest(config, args)
    else:
        build_live_loop(config, dry_run=args.dry_run).run()
    return 0

if __name__=='__main__':
    raise SystemExit(main())
