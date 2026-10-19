"""Synthetic broker-independent risk gates and sticky breaker scope tests."""
from dataclasses import replace
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from core.regime_strategies import Signal, Direction
from core.risk_manager import RiskManager,RiskConfig,PortfolioState,CircuitBreaker,TradingState

@pytest.fixture
def risk(tmp_path):
    return RiskManager(RiskConfig(lock_path=str(tmp_path/'halt.lock')))

@pytest.fixture
def account():
    return PortfolioState(100000,100000,100000,timestamp=datetime(2025,1,6))

@pytest.fixture
def signal():
    return Signal('SPY',Direction.LONG,.9,100,90,None,.95,1.25,0,'BULL',.9,datetime(2025,1,6),'test','LowVolBullStrategy',{'vol_regime':'low','overnight':False,'sector':'tech'})

@pytest.mark.parametrize('kind,dd,expected',[
    ('daily',.021,TradingState.REDUCED),('daily',.031,TradingState.HALTED),
    ('weekly',.051,TradingState.REDUCED),('weekly',.071,TradingState.HALTED),
    ('peak',.101,TradingState.KILLED)])
def test_each_breaker_actual_pnl(risk,kind,dd,expected):
    state=risk.breaker.check(**{kind+'_dd':dd},equity=100000,positions=['SPY'],regime='wrong_model')
    assert state==expected
    event=risk.breaker.get_history()[-1]
    assert event['drawdown']==dd and event['equity']==100000
    assert event['regime']=='wrong_model'
    assert event['positions_closed']==(['SPY'] if expected in (TradingState.HALTED,TradingState.KILLED) else [])
    assert risk.breaker.check()==expected


def test_lock_create_block_manual_delete(risk,account,signal):
    risk.breaker.check(peak_dd=.11)
    path=__import__('pathlib').Path(risk.config.lock_path)
    assert path.exists()
    fresh=RiskManager(risk.config)
    assert not fresh.validate_signal(signal,account).approved
    path.unlink()
    assert fresh.validate_signal(signal,account).approved


def test_lock_manual_delete_resets_acknowledged_peak(risk,account,signal):
    account.peak_equity = 120000
    risk.update_equity(120000,account.timestamp)
    risk.update_equity(100000,account.timestamp)
    assert risk.evaluate_state(100000)==TradingState.KILLED
    __import__('pathlib').Path(risk.config.lock_path).unlink()
    account.timestamp += timedelta(days=7)
    assert risk.validate_signal(signal,account).approved


def test_sticky_breakers_and_resets(risk):
    b=risk.breaker
    b.check(daily_dd=.021,weekly_dd=.051)
    b.reset_daily()
    assert b.check()==TradingState.REDUCED
    b.reset_weekly()
    assert b.check()==TradingState.NORMAL
    assert b.check(daily_dd=.02)==TradingState.NORMAL


def test_calendar_reset_and_overnight_loss(risk):
    t=datetime(2025,1,6)
    risk.update_equity(100000,t)
    risk.update_equity(97000,t+timedelta(days=1))
    assert risk.daily_drawdown(97000)==pytest.approx(.03)
    risk.evaluate_state(97000)
    assert risk.state==TradingState.REDUCED # strict >3%, floating point tolerance tested separately
    risk.record_trade()
    risk.update_equity(97000,t+timedelta(days=7))
    assert risk.trades_today==0
    assert risk.evaluate_state(97000)==TradingState.NORMAL


def test_missing_stop(risk,account,signal):
    assert risk.validate_signal(replace(signal,stop_loss=None),account).rejection_reason=='missing_stop_loss'


def test_formula_caps_minimum_gap(risk):
    assert risk.position_size(100000,100,90)==100
    assert risk.position_size(100000,100,99)==150
    assert risk.position_size(100000,100,90,regime_max=.05)==50
    assert risk.position_size(500,100,90)==0
    q=risk.position_size(100000,100,90,overnight=True)
    assert q==66 and q*10*3<=2000
    assert risk.position_size(100000,100,100)==0

@pytest.mark.parametrize('condition',['mid','uncertain','low_probability','breaker','account_breaker','positions','flicker'])
def test_each_leverage_force(risk,account,signal,condition):
    if condition=='mid': signal=replace(signal,metadata={'vol_regime':'mid'})
    if condition=='uncertain': signal=replace(signal,metadata={'vol_regime':'low','uncertain':True})
    if condition=='low_probability': signal=replace(signal,confidence=.4)
    if condition=='breaker': risk.state=TradingState.REDUCED
    if condition=='account_breaker': account.circuit_breaker_status='reduced'
    if condition=='positions': account.positions={s:{'market_value':100} for s in ['A','B','C']}
    if condition=='flicker': account.flicker_rate=5
    assert risk.leverage(signal,account)==1.


def test_low_vol_leverage_and_single_cap(risk,signal,account):
    assert risk.leverage(signal,account)==1.25
    decision=risk.validate_signal(replace(signal,stop_loss=99),account)
    assert decision.approved and decision.quantity==150

@pytest.mark.parametrize('cap',['sector','exposure','concurrent','daily','buying_power','spread','tradeable'])
def test_order_caps(risk,account,signal,cap):
    if cap=='sector': account.positions={'A':{'market_value':30000,'sector':'tech'}}
    if cap=='exposure': account.positions={'A':{'market_value':80000,'sector':'other'}}
    if cap=='concurrent': account.positions={str(i):{'market_value':100,'sector':'other'} for i in range(5)}
    if cap=='daily': account.daily_trades=20
    if cap=='buying_power': account.buying_power=1
    quote={'bid':99,'ask':101} if cap=='spread' else {'tradeable':False} if cap=='tradeable' else None
    assert not risk.validate_signal(signal,account,quote).approved


def test_daily_trade_counter(risk):
    for _ in range(20): risk.record_trade()
    assert not risk.can_trade()
    risk.reset_daily()
    assert risk.can_trade()


def test_duplicate_side_and_timeout(risk,signal,account):
    assert risk.validate_signal(signal,account).approved
    assert risk.validate_signal(signal,account).rejection_reason=='duplicate'
    account.timestamp+=timedelta(seconds=60)
    assert risk.validate_signal(signal,account).approved

@pytest.mark.parametrize('correlation,expected',[(.8,50),(.9,0)])
def test_rolling_correlation(risk,signal,account,correlation,expected):
    rng=np.random.default_rng(33)
    x=rng.normal(size=60); x=(x-x.mean())/x.std()
    z=rng.normal(size=60); z-=z.mean(); z-=x*(z@x)/(x@x); z/=z.std()
    y=correlation*x+np.sqrt(1-correlation**2)*z
    account.returns=pd.DataFrame({'SPY':x,'A':y})
    account.positions={'A':{'market_value':1000,'sector':'other'}}
    d=risk.validate_signal(signal,account)
    assert d.quantity==expected and d.approved==bool(expected)


def test_reduced_signal_size(risk,signal,account):
    account.daily_pnl=-2200
    d=risk.validate_signal(signal,account)
    assert d.approved and d.quantity==50 and d.modified_signal.leverage==1


def test_halt_requests_all_closures(risk,signal,account):
    account.daily_pnl=-4000
    account.positions={'A':{'market_value':1000}}
    d=risk.validate_signal(signal,account)
    assert not d.approved and d.close_positions==['A']


def test_breaker_update_actual_currency_pnl(risk,account):
    account.daily_pnl=-4000
    assert risk.breaker.update(account,regime='BULL')==TradingState.HALTED
    assert risk.breaker.get_history()[-1]['drawdown']==pytest.approx(4000/104000)
