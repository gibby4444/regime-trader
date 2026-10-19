"""Offline simulation entry point; historical downloading is explicitly requested by CLI."""
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
    parser.add_argument('mode',choices=['backtest','stress'])
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

def main(argv=None):
    args=parse_args(argv)
    run_backtest(load_config(args.config),args)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
