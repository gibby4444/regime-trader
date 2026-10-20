"""Synthetic execution, causal walk-forward training, analytics and stress tests."""
from dataclasses import replace
import socket
import numpy as np
import pandas as pd
import pytest
from backtest.backtester import BacktestConfig,WalkForwardBacktester
from backtest.performance import PerformanceReport,max_drawdown,underwater_duration,confidence_breakdown
from backtest.stress_test import StressTester,StressScenario
from core.hmm_engine import RegimeInfo,RegimeState
from core.regime_strategies import StrategyConfig,StrategyOrchestrator
from core.risk_manager import RiskManager,RiskConfig
from tests.conftest import make_synthetic_bars

INFO=[RegimeInfo(i,name,0,vol,strategy,1.25,.95,.55) for i,name,vol,strategy in
      [(0,'BULL',.01,'LowVolBullStrategy'),(1,'NEUTRAL',.02,'MidVolCautiousStrategy'),(2,'CRASH',.03,'HighVolDefensiveStrategy')]]

class TinyFeatures:
    def build_features(self,bars):
        return bars[['close']].pct_change().dropna()
    def returns(self,bars):
        return np.log(bars.close/bars.close.shift())

class ForwardOnlyHMM:
    fitted=[]
    def fit(self,features,returns,training_date):
        assert features.index[-1]==training_date
        self.fitted.append(features.index.copy())
        self.regime_info=INFO
        return self
    def run_filter(self,features):
        self.previous=features.iloc[-1,0]
    def update(self,row,ts,log_events=False):
        self.previous=float(row.iloc[0])
        return RegimeState('BULL',0,.9,np.array([.9,.05,.05]),ts,True,10)
    def is_flickering(self):
        return False

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Tests must never use network')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(socket,'create_connection',forbidden)

@pytest.fixture
def bars():
    b=make_synthetic_bars(100,seed=42)[0]
    # Calm upward series with realistic next-open marks; distant stops stay live.
    close=100*np.exp(np.arange(100)*.0008)
    b['close']=close; b['open']=np.r_[100,close[:-1]]
    b['high']=b[['open','close']].max(axis=1)+1
    b['low']=b[['open','close']].min(axis=1)-1
    return {'SPY':b}

@pytest.fixture
def backtester(tmp_path):
    return WalkForwardBacktester(BacktestConfig(train_window=12,test_window=10,step_size=10,random_seeds=2,stress_simulations=2,stress_points=2),
        ForwardOnlyHMM(),StrategyOrchestrator(StrategyConfig(ema_window=4,atr_window=3),INFO),
        RiskManager(RiskConfig(lock_path=str(tmp_path/'live.lock'))),TinyFeatures())


def test_walk_forward_training_only_and_partial_folds(backtester,bars):
    ForwardOnlyHMM.fitted=[]
    result=backtester.run(bars)
    for train,fold in zip(ForwardOnlyHMM.fitted,result.folds):
        assert len(train)==12 and train.max()<fold.test_start
    assert result.equity_curve.index.is_unique
    assert result.equity_curve.index[-1]==bars['SPY'].index[-1]
    assert (result.weights.sum(axis=1)<=.80+1e-8).all()
    first_fill=result.trades.iloc[0].timestamp
    assert result.weights.loc[first_fill,'SPY']<=.15 # caps constrain orders; weights drift between rebalances


def test_results_prefix_invariant_and_future_perturbation(backtester,bars):
    full=backtester.run(bars)
    cut=bars['SPY'].index[56]
    short=backtester.run({'SPY':bars['SPY'].loc[:cut]})
    pd.testing.assert_series_equal(short.equity_curve,full.equity_curve.loc[:cut])
    pd.testing.assert_frame_equal(short.trades,full.trades[full.trades.timestamp<=cut].reset_index(drop=True))
    changed=bars['SPY'].copy()
    changed.loc[changed.index>cut,['open','high','low','close']]*=.5
    perturbed=backtester.run({'SPY':changed})
    pd.testing.assert_series_equal(short.equity_curve,perturbed.equity_curve.loc[:cut])


def test_one_bar_delay_slippage_and_accounting(backtester,bars):
    result=backtester.run(bars)
    first=result.trades.iloc[0]
    first_oos=result.folds[0].test_start
    assert first.timestamp==bars['SPY'].index[bars['SPY'].index.get_loc(first_oos)+1]
    assert first.price==pytest.approx(bars['SPY'].loc[first.timestamp,'open']*1.0005)
    cash=100000-first.quantity*first.price
    assert result.equity_curve.loc[first.timestamp]==pytest.approx(cash+first.quantity*bars['SPY'].loc[first.timestamp,'close'])
    assert len(result.trades)==1 # unchanged allocation does not churn


def test_overlap_stitches_once(backtester,bars):
    backtester.config=replace(backtester.config,step_size=5)
    result=backtester.run(bars)
    assert result.equity_curve.index.is_unique
    assert len(result.equity_curve)==len(bars['SPY'])-13


def test_gap_stop_and_breaker_liquidation(backtester,bars):
    b=bars['SPY'].copy()
    b.iloc[25:, :4]*=.5
    result=backtester.run({'SPY':b})
    exit=result.trades[result.trades.side=='sell'].iloc[0]
    assert exit.timestamp==b.index[25]
    assert exit.price==pytest.approx(b.iloc[25].open*.9995)
    assert result.breaker_history
    assert not __import__('pathlib').Path(backtester.risk.config.lock_path).exists()
    pd.testing.assert_series_equal(backtester.run(bars).equity_curve,backtester.run(bars).equity_curve)


def test_benchmarks_and_seed_counts(backtester,bars):
    results,random=backtester.compare(bars,seeds=2)
    assert set(results)=={'buy_hold','sma200'}
    assert not results['buy_hold'].trades.empty
    assert results['sma200'].trades.empty # fewer than 200 past bars
    assert {'mean','std'}==set(random)
    assert np.isfinite(random.loc['total_return']).all()


def test_metrics_known_values_and_confidence():
    idx=pd.bdate_range('2025-01-01',periods=5)
    eq=pd.Series([100,110,99,100,120],index=idx)
    assert max_drawdown(eq)==pytest.approx(.1)
    assert underwater_duration(eq)==2
    trades=pd.DataFrame({'side':['sell']*4,'pnl':[10,-5,20,-10],'quantity':[1]*4,'price':[100]*4,
                         'confidence':[.4,.55,.65,.9],'regime':['BULL']*4,'holding_days':[1,2,3,4]})
    report=PerformanceReport.from_backtest(eq,pd.Series('BULL',index=idx),trades=trades)
    assert report.total_return==pytest.approx(.2)
    assert report.win_rate==.5 and report.profit_factor==2
    assert report.max_drawdown_duration==2 and report.total_trades==4
    assert list(report.confidence_stats.trades)==[1,1,1,1]
    assert report.regime_stats.loc['BULL','time_pct']==1


def test_stress_synthetic_counts_and_inputs_unchanged(backtester,bars):
    stress=StressTester(backtester,seed=5)
    original=bars['SPY'].copy()
    injected=stress.simulate_gap(original,-.1,original.index[20])
    pd.testing.assert_frame_equal(injected.iloc[:20],original.iloc[:20])
    assert injected.iloc[20].open==pytest.approx(original.iloc[20].open*.9)
    assert (injected.high>=injected[['open','close']].max(axis=1)).all()
    results=stress.run_all(bars,simulations=2,points=2)
    assert len(results)==6
    assert set(results.scenario)=={'crash','overnight_gap','regime_shuffle'}
    pd.testing.assert_frame_equal(original,bars['SPY'])


def test_csv_output(backtester,bars,tmp_path):
    backtester.run(bars).to_csv(tmp_path)
    assert all((tmp_path/f).exists() for f in ['equity_curve.csv','trade_log.csv','regime_history.csv'])


def test_cli_and_settings_are_offline():
    from main import load_config,parse_args,build_backtester
    cfg=load_config()
    args=parse_args(['backtest','--symbols','SPY','--start','2020-01-01','--compare','--stress-test'])
    assert args.symbols==['SPY'] and args.compare and args.stress_test
    bt=build_backtester(cfg)
    assert bt.config.train_window==252 and bt.hmm.config.min_train_bars==252
    assert cfg['broker']['paper_trading'] is True and cfg['risk']['max_exposure']==.8


def test_real_hmm_cli_pipeline_synthetic(monkeypatch,tmp_path):
    """Exercise actual fitting, filtered inference, reporting, comparisons and stress CLI."""
    from main import load_config,run_backtest,parse_args
    from data import market_data
    cfg=load_config()
    cfg['hmm'].update(n_candidates=[3],n_init=1,n_iter=12,min_train_bars=30,covariance_type='diag')
    cfg['features'].update(return_windows=[1,2,3],vol_window=4,vol_ratio_short=2,volume_window=4,
                           volume_sma_window=3,slope_lookback=2,adx_window=3,trend_sma_window=5,
                           rsi_window=3,long_sma_window=6,roc_windows=[2,3],atr_window=3,zscore_window=5)
    cfg['strategy'].update(ema_window=5,atr_window=3)
    cfg['backtest'].update(train_window=30,test_window=50,step_size=50,random_seeds=1,stress_simulations=1,stress_points=1)
    data={'SPY':make_synthetic_bars(100,seed=17)[0]}
    monkeypatch.setattr(market_data,'load_yfinance_history',lambda *args:data)
    args=parse_args(['backtest','--symbols','SPY','--compare','--stress-test','--output',str(tmp_path)])
    result=run_backtest(cfg,args)
    assert len(result.equity_curve)>0 and result.regime_history.confidence.between(0,1).all()
    assert (tmp_path/'stress_results.csv').exists() and (tmp_path/'random_benchmark.csv').exists()


def test_daily_close_halt_survives_session_reset_until_liquidation(backtester,bars):
    backtester.risk.config.daily_dd_halt=.0001
    backtester.risk.config.daily_dd_reduce=.00005
    b=bars['SPY'].copy()
    b.loc[b.index[25],'close']*=.99
    b.loc[b.index[25],'low']=b.loc[b.index[25],'close']-.1
    result=backtester.run({'SPY':b})
    exits=result.trades[(result.trades.reason=='circuit_breaker')&(result.trades.timestamp==b.index[26])]
    assert not exits.empty


@pytest.mark.parametrize('stop,entry_open,expected', [
    (100, 100, 'reject'), (101, 100, 'reject'), (None, 100, 'reject'),
    (99, 98, 'reject'),  # valid at signal close, invalid after next-open gap
    (100.02, 100, 'reject'),  # below slipped fill, still above executable open
    (99.5, 100, 'stop'), (98, 100, 'hold'),
])
def test_entry_stop_ordering(backtester, bars, monkeypatch, stop, entry_open, expected):
    b = bars['SPY'].copy()
    b[['open', 'close']] = 100.
    b['high'], b['low'] = 101., 99.
    first_fill = b.index[14]
    b.loc[first_fill, 'open'] = entry_open
    b.loc[first_fill, 'low'] = min(99., entry_open - 1)
    original = StrategyOrchestrator.generate_signals

    def signals(self, *args, **kwargs):
        return [replace(s, stop_loss=stop) for s in original(self, *args, **kwargs)
                if s.timestamp == b.index[13]]

    monkeypatch.setattr(StrategyOrchestrator, 'generate_signals', signals)
    result = backtester.run({'SPY': b})
    if expected == 'reject':
        assert result.trades.empty
    elif expected == 'stop':
        assert list(result.trades.side) == ['buy', 'sell']
        assert (result.trades.timestamp == first_fill).all()
        assert result.trades.iloc[-1].price == pytest.approx(stop * .9995)
    else:
        assert list(result.trades.side) == ['buy']


@pytest.mark.parametrize('add', [False, True])
def test_standing_stop_changes_only_on_fill_and_never_widens(backtester, bars, monkeypatch, add):
    b = bars['SPY'].copy()
    b[['open', 'close']] = 100.
    b['high'], b['low'] = 101., 99.
    original = StrategyOrchestrator.generate_signals

    def signals(self, *args, **kwargs):
        result = []
        for s in original(self, *args, **kwargs):
            if s.timestamp not in (b.index[13], b.index[14]):
                continue
            first = s.timestamp == b.index[13]
            result.append(replace(s, stop_loss=98 if first else (97 if add else 99.5),
                                  metadata={**s.metadata, 'target_weight': .05 if first and add else .15}))
        return result

    monkeypatch.setattr(StrategyOrchestrator, 'generate_signals', signals)
    monkeypatch.setattr(StrategyOrchestrator, 'needs_rebalance', lambda *args: True)
    if add:
        b.loc[b.index[16], 'low'] = 97.5
    result = backtester.run({'SPY': b})
    if add:
        assert list(result.trades.side) == ['buy', 'buy', 'sell']
        assert result.trades.iloc[-1].price == pytest.approx(98 * .9995)
    else:
        assert list(result.trades.side) == ['buy']
