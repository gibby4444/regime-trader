"""Step 9: real decision components, synthetic bars, mocked broker transports."""
from copy import deepcopy
from dataclasses import fields, replace
import os
import socket
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import Direction, Signal, StrategyConfig, StrategyOrchestrator
from core.risk_manager import PortfolioState, RiskConfig, RiskManager
from core.trading_loop import TradingLoop
from data.feature_engineering import FeatureConfig, FeatureEngineer
from main import build_backtester, load_config, parse_args
from tests.conftest import make_synthetic_bars


@pytest.fixture(autouse=True)
def offline(monkeypatch, request):
    if request.node.name == 'test_real_alpaca_paper_connectivity':
        return
    def forbidden(*args, **kwargs):
        raise AssertionError('Integration tests prohibit network, SDK connections and credentials')
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    for path in ('broker.alpaca_client.load_dotenv', 'broker.alpaca_client.TradingClient',
                 'broker.alpaca_client.StockHistoricalDataClient',
                 'broker.position_tracker.TradingStream', 'data.market_data.StockDataStream',
                 'data.market_data.load_yfinance_history'):
        monkeypatch.setattr(path, forbidden)
    monkeypatch.setattr(AlpacaClient, 'connect', forbidden)


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg['broker']['symbols'] = ['SPY']
    cfg['hmm'].update(n_candidates=[3], n_init=1, n_iter=20, min_train_bars=40,
                      covariance_type='diag')
    cfg['features'].update(return_windows=[1, 2, 3], vol_window=4, vol_ratio_short=2,
                           volume_window=4, volume_sma_window=3, slope_lookback=2,
                           adx_window=3, trend_sma_window=5, rsi_window=3,
                           long_sma_window=6, roc_windows=[2, 3], atr_window=3,
                           zscore_window=5)
    cfg['strategy'].update(ema_window=5, atr_window=3)
    cfg['backtest'].update(train_window=40, test_window=25, step_size=25)
    cfg['risk']['lock_path'] = str(tmp_path/'halt.lock')
    return cfg


@pytest.fixture
def pipeline(config, tmp_path):
    bars = make_synthetic_bars(150, seed=17)[0]
    features = FeatureEngineer(FeatureConfig.from_dict(config['features']))
    matrix = features.build_features(bars.iloc[:80])
    engine = HMMEngine(HMMConfig.from_dict(config['hmm'])).fit(
        matrix, features.returns(bars).reindex(matrix.index))
    engine.run_filter(matrix)
    client = MagicMock(spec=AlpacaClient)
    client.trade_context = {}
    client.trading = MagicMock()
    client.get_account.return_value = dict(equity=100000, cash=100000, buying_power=100000)
    client.get_positions.return_value = []
    client.get_orders.return_value = []
    client.get_clock.return_value = dict(is_open=True)

    def make_loop(dry_run):
        risk = RiskManager(RiskConfig.from_dict(config['risk']))
        tracker = PositionTracker(client, risk, PortfolioState(100000, 100000, 100000))
        executor = OrderExecutor(client, risk, tracker.portfolio)
        loop = TradingLoop(config, client=client, feed=MagicMock(), engine=deepcopy(engine),
                           features=features, strategy=StrategyOrchestrator(
                               StrategyConfig.from_dict(config['strategy'], config['hmm']), engine.regime_info),
                           risk=risk, tracker=tracker, executor=executor, dashboard=MagicMock(),
                           alerts=MagicMock(), train=MagicMock(), dry_run=dry_run,
                           now=lambda: pd.Timestamp('2026-10-19T16:00:00Z').to_pydatetime(),
                           model_path=tmp_path/'hmm.pkl', snapshot_path=tmp_path/'state_snapshot.json',
                           output=lambda _: None)
        loop.trained_at = loop.now()
        loop.bars = {'SPY': bars.iloc[:80].copy()}
        return loop
    return bars, client, make_loop


def test_end_to_end_dry_run_and_simulated_orders(pipeline, config):
    bars, client, make_loop = pipeline
    loop = make_loop(True)
    for _, row in bars.iloc[80:90].iterrows():
        loop.process_bar('SPY', row)
    assert loop.last_regime is not None
    assert loop.counts['approved'] + loop.counts['modified'] > 0
    client.trading.submit_order.assert_not_called()
    client.trading.replace_order_by_id.assert_not_called()
    client.trading.cancel_order_by_id.assert_not_called()
    # The backtester fills simulated orders through the same real feature/HMM/
    # strategy/risk layers; dry-run itself deliberately never submits orders.
    result = build_backtester(config).run({'SPY': bars})
    assert not result.trades.empty
    assert np.isfinite(result.equity_curve).all()
    assert (result.trades.quantity > 0).all()
    client.trading.submit_order.assert_not_called()


def test_full_backtest_identical_with_different_end_dates(config):
    bars = make_synthetic_bars(150, seed=17)[0]
    cutoff = bars.index[112]  # Ends inside an OOS fold.
    full = build_backtester(config).run({'SPY': bars})
    short = build_backtester(config).run({'SPY': bars.loc[:cutoff]})
    assert not short.trades.empty
    for name in ('equity_curve', 'returns', 'regimes'):
        pd.testing.assert_series_equal(getattr(short, name), getattr(full, name).loc[:cutoff], check_exact=True)
    for name in ('weights', 'regime_history'):
        pd.testing.assert_frame_equal(getattr(short, name), getattr(full, name).loc[:cutoff], check_exact=True)
    pd.testing.assert_frame_equal(short.trades,
        full.trades.loc[full.trades.timestamp <= cutoff].reset_index(drop=True), check_exact=True)


def proposed():
    return Signal('SPY', Direction.LONG, .99, 100, 99, 110, 1000, 1000,
                  0, 'calm', .99, pd.Timestamp('2026-10-19T16:00:00Z'), 'stress', 'test',
                  {'vol_regime': 'low', 'target_weight': 1e6, 'overnight': False})


def test_risk_stress_caps_duplicates_and_missing_stop(config):
    risk = RiskManager(RiskConfig.from_dict(config['risk']))
    account = PortfolioState(100000, 100000, 100000, timestamp=proposed().timestamp,
                             positions={'QQQ': {'market_value': 75000, 'sector': 'other'}})
    decision = risk.validate_signal(proposed(), account)
    assert decision.approved
    assert decision.quantity * 100 <= 100000 * risk.config.max_single_position
    assert decision.quantity * 100 + 75000 <= 100000 * .80
    assert decision.quantity * (100-99) <= 100000 * risk.config.max_risk_per_trade
    assert decision.modified_signal.leverage <= risk.config.max_leverage
    assert decision.modifications
    for _ in range(10):
        assert risk.validate_signal(proposed(), account).rejection_reason == 'duplicate'
    assert risk.validate_signal(replace(proposed(), stop_loss=None), account).rejection_reason == 'missing_stop_loss'


def test_mocked_alpaca_paper_bracket_tighten_cancel_clean_state(pipeline):
    _, client, make_loop = pipeline
    loop = make_loop(False)
    orders, positions = [], []
    client.get_orders.side_effect = lambda: deepcopy(orders)
    client.get_positions.side_effect = lambda: deepcopy(positions)

    def submit(*, order_data):
        req = order_data
        response = dict(id='entry' if req.side.value == 'buy' else 'exit',
                        client_order_id=req.client_order_id, symbol=req.symbol,
                        status='filled', filled_qty=req.qty, filled_avg_price=100)
        if req.side.value == 'buy':
            assert req.order_class.value == 'bracket'
            assert req.stop_loss.stop_price == 95 and req.take_profit.limit_price == 110
            positions.append(dict(symbol='SPY', qty=req.qty, avg_entry_price=100, current_price=100))
            orders.append(dict(id='entry', symbol='SPY', side='buy', legs=[
                dict(id='stop', symbol='SPY', side='sell', stop_price=95),
                dict(id='profit', symbol='SPY', side='sell', limit_price=110)]))
        else:
            positions.clear()
        return response

    client.trading.submit_order.side_effect = submit
    client.call.side_effect = lambda *a, **kw: dict(id='entry', symbol='SPY', status='filled', filled_qty=100)
    def tighten(order_id, *, order_data):
        assert order_id == 'stop'
        orders[0]['legs'][0]['stop_price'] = order_data.stop_price
        return dict(id='stop', symbol='SPY', status='new', filled_qty=0)
    client.trading.replace_order_by_id.side_effect = tighten
    client.trading.cancel_order_by_id.side_effect = lambda _: orders.clear()
    sig = replace(proposed(), position_size_pct=.1, leverage=1, stop_loss=95,
                  metadata={'target_weight': .1})
    result = loop.executor.submit_bracket_order(sig)
    assert result.status == 'filled'
    loop.tracker.sync()
    assert loop.tracker.positions['SPY'].stop == 95
    loop.executor.modify_stop('SPY', 97)
    loop.tracker.sync()
    assert loop.tracker.positions['SPY'].stop == 97
    with pytest.raises(ValueError, match='only tighten'):
        loop.executor.modify_stop('SPY', 96)
    assert client.trading.replace_order_by_id.call_count == 1
    loop.executor.cancel_order('entry')
    loop.executor.close_position('SPY')
    loop.tracker.sync()
    assert client.get_orders() == [] and client.get_positions() == []
    assert loop.tracker.positions == {} and loop.portfolio.positions == {}


def test_kill_mid_submission_restart_from_snapshot_no_double_entry(pipeline):
    bars, client, make_loop = pipeline
    # A rebound keeps the real strategy's EMA protective stop below entry.
    bars.loc[bars.index[80:82], 'close'] = 110
    bars.loc[bars.index[80:82], 'high'] = 111
    original = make_loop(False)
    def killed(*args, **kwargs):
        # Broker accepted, then the process disappeared before a response.
        raise SystemExit('simulated abrupt kill')
    client.trading.submit_order.side_effect = killed
    with pytest.raises(SystemExit, match='abrupt kill'):
        original.process_bar('SPY', bars.iloc[80])
    assert original.snapshot_path.name == 'state_snapshot.json'
    assert original.snapshot_path.exists()
    restarted = make_loop(False)
    restarted.tracker.start = MagicMock()  # No stream threads or sockets.
    assert restarted.startup()  # Reconcile broker, restore disk, resume session.
    assert restarted.risk._duplicates == original.risk._duplicates
    assert restarted.risk.peak_equity == original.risk.peak_equity
    assert restarted.counts == original.counts
    assert restarted.last_prediction == original.last_prediction
    np.testing.assert_array_equal(restarted.engine._log_alpha, original.engine._log_alpha)
    assert vars(restarted.engine.stability) == vars(original.engine.stability)
    restarted.process_bar('SPY', bars.iloc[80])  # Replay consumed bar.
    restarted.process_bar('SPY', bars.iloc[81])  # New signal inside duplicate window.
    assert restarted.counts['rejected'] > 0
    assert client.trading.submit_order.call_count == 1


@pytest.mark.parametrize('argv,mode,flags', [
    (['train-only'], 'train-only', {}),
    (['backtest', '--symbols', 'SPY', '--start', '2020-01-01', '--end', '2024-12-31'],
     'backtest', {'symbols': ['SPY'], 'start': '2020-01-01', 'end': '2024-12-31'}),
    (['backtest', '--compare'], 'backtest', {'compare': True}),
    (['backtest', '--stress-test'], 'backtest', {'stress_test': True}),
    (['live', '--dry-run'], 'live', {'dry_run': True}),
    (['live'], 'live', {'dry_run': False}),
])
def test_l12_cli_parse_only(argv, mode, flags, monkeypatch):
    monkeypatch.setattr('main.build_live_loop', lambda *a, **kw: pytest.fail('parse must not execute'))
    args = parse_args(argv)
    assert args.mode == mode
    for key, value in flags.items():
        assert getattr(args, key) == value


def test_settings_complete_and_paper_defaults():
    cfg = load_config()
    for section, cls in [('hmm', HMMConfig), ('features', FeatureConfig), ('risk', RiskConfig)]:
        assert set(cfg[section]) == {f.name for f in fields(cls)}
    from backtest.backtester import BacktestConfig
    assert set(cfg['backtest']) == {f.name for f in fields(BacktestConfig)}
    assert set(cfg['strategy']) == {f.name for f in fields(StrategyConfig)} - {'min_confidence', 'transition_size_mult'}
    build_backtester(cfg)
    assert cfg['broker']['paper_trading'] is True
    assert cfg['risk']['max_exposure'] == .80
    assert cfg['monitoring']['email']['enabled'] is False
    assert cfg['monitoring']['webhook']['enabled'] is False


@pytest.mark.skipif(os.getenv('RUN_ALPACA_PAPER') != '1', reason='Opt-in real paper connectivity; never run in offline verification')
def test_real_alpaca_paper_connectivity():
    """Explicit opt-in only: paper account read, no orders or live-capital switch."""
    client = AlpacaClient({'paper_trading': True})
    try:
        client.connect()
        assert client.paper and client.base_url == AlpacaClient.PAPER_URL
        assert float(client.get_account()['equity']) > 0
    finally:
        client.disconnect()
