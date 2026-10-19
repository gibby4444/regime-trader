"""Causal allocation simulation with rolling IS fitting and next-open execution."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import copy
import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from core.risk_manager import RiskManager, PortfolioState, TradingState
from data.feature_engineering import FeatureEngineer, validate_bars

@dataclass
class BacktestConfig:
    slippage_pct: float = .0005
    initial_capital: float = 100000
    train_window: int = 252
    test_window: int = 126
    step_size: int = 126
    risk_free_rate: float = .045
    random_seeds: int = 100
    stress_simulations: int = 100
    stress_points: int = 10
    commission: float = 0

    @classmethod
    def from_dict(cls, cfg):
        return cls(**cfg)

@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    regimes: pd.Series
    trades: pd.DataFrame
    folds: list[Fold] = field(default_factory=list)
    regime_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_capital: float = 100000
    breaker_history: list[dict] = field(default_factory=list)

    def to_csv(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.equity_curve.to_csv(path/'equity_curve.csv')
        self.trades.to_csv(path/'trade_log.csv', index=False)
        self.regime_history.to_csv(path/'regime_history.csv')

class WalkForwardBacktester:
    def __init__(self, config: BacktestConfig, hmm: HMMEngine, strategy: StrategyOrchestrator,
                 risk: RiskManager, features: FeatureEngineer):
        self.config, self.hmm, self.strategy, self.risk, self.features = config, hmm, strategy, risk, features
        if min(config.train_window, config.test_window, config.step_size) <= 0:
            raise ValueError('Windows must be positive')
        if config.step_size > config.test_window:
            raise ValueError('step_size cannot leave unevaluated gaps')

    def make_folds(self, index):
        c = self.config
        return [Fold(index[i-c.train_window], index[i-1], index[i], index[min(i+c.test_window,len(index))-1])
                for i in range(c.train_window, len(index), c.step_size)]

    def apply_slippage(self, price, side):
        return price*(1+self.config.slippage_pct if side.lower() == 'buy' else 1-self.config.slippage_pct)

    def run(self, bars_by_symbol, *, mode='strategy', seed=0, shuffle_labels=False):
        """Fresh independent account each run. Simulation locks never affect live locks.

        Folds count valid feature observations, after causal indicator warm-up.
        Overlapping OOS segments use the newest available fitted model exactly once.
        """
        bars = {s:validate_bars(b).copy() for s,b in bars_by_symbol.items()}
        if not bars:
            raise ValueError('At least one symbol required')
        index = next(iter(bars.values())).index
        for b in bars.values():
            index = index.intersection(b.index)
        bars = {s:b.loc[index] for s,b in bars.items()}
        primary = next(iter(bars))
        features = self.features.build_features(bars[primary])
        folds = self.make_folds(features.index)
        if not folds:
            raise ValueError('Insufficient valid features for an IS window and OOS bar')
        with TemporaryDirectory(prefix='regime-backtest-') as directory:
            risk = RiskManager(replace(self.risk.config, lock_path=str(Path(directory)/'trading_halted.lock')))
            return self._simulate(bars, features, folds, risk, mode, seed, shuffle_labels)

    def _simulate(self, bars, features, folds, risk, mode, seed, shuffle_labels):
        rng = np.random.default_rng(seed)
        primary = next(iter(bars))
        c = self.config
        cash = c.initial_capital
        shares = {s:0 for s in bars}
        stops, costs, opened, contexts = {}, {}, {}, {}
        pending = []
        pending_liquidation = False
        records, trades, history, weight_rows = [], [], [], []
        engine = copy.deepcopy(self.hmm)
        strategy = copy.deepcopy(self.strategy)
        starts = {f.test_start:f for f in folds}
        last_regime = None
        sector = {s:'unknown' for s in bars}
        close_panel = pd.DataFrame({s:b.close for s,b in bars.items()})
        returns = close_panel.pct_change(fill_method=None)

        def mark(ts, column):
            return cash+sum(shares[s]*float(bars[s].loc[ts,column]) for s in bars)

        def execute(s, target, price, ts, reason, context=None):
            nonlocal cash
            delta = int(target)-shares[s]
            if not delta:
                return
            fill = self.apply_slippage(price, 'buy' if delta>0 else 'sell')
            pnl = 0.
            holding = 0
            old = shares[s]
            if delta>0:
                costs[s] = (costs.get(s,0)*old+delta*fill)/target
                if not old:
                    opened[s] = ts
                contexts[s] = context or {}
            else:
                pnl = -delta*(fill-costs.get(s,fill))-c.commission
                holding = (ts-opened.get(s,ts)).days
            cash -= delta*fill+c.commission
            shares[s] = int(target)
            risk.record_trade()
            ctx = context or contexts.get(s,{})
            trades.append(dict(timestamp=ts, symbol=s, side='buy' if delta>0 else 'sell',
                               quantity=abs(delta), price=fill, pnl=pnl, holding_days=holding,
                               regime=ctx.get('regime',last_regime), confidence=ctx.get('confidence',0), reason=reason))

        for ts in features.loc[folds[0].test_start:].index:
            if ts in starts and mode == 'strategy':
                fold = starts[ts]
                train = features.loc[fold.train_start:fold.train_end]
                engine.fit(train, self.features.returns(bars[primary]).reindex(train.index), training_date=fold.train_end)
                strategy.update_regimes(engine.regime_info)
                engine.run_filter(train)
                label_permutation=rng.permutation(len(engine.regime_info))
            if mode in ('buy_hold','sma200') and hasattr(risk, '_benchmark_next') and (mode != 'buy_hold' or not any(shares.values())):
                eq_open=mark(ts,'open')
                for s,w in risk._benchmark_next.items():
                    execute(s,int(eq_open*w/float(bars[s].loc[ts,'open'])),float(bars[s].loc[ts,'open']),ts,'benchmark')
            # P&L marked at this open includes overnight gaps, before any new order.
            equity = mark(ts,'open')
            risk.update_equity(equity, ts)
            state = risk.evaluate_state(equity, [s for s in shares if shares[s]], last_regime)
            if (pending_liquidation or state in (TradingState.HALTED, TradingState.KILLED)) and mode not in ('buy_hold','sma200'):
                for s in shares:
                    execute(s,0,float(bars[s].loc[ts,'open']),ts,'circuit_breaker')
                pending = []
                pending_liquidation = False
            else:
                # Stops are standing orders: gap-through fills at open, otherwise at stop.
                stopped = set()
                for s in shares:
                    if mode not in ('buy_hold','sma200') and shares[s] and float(bars[s].loc[ts,'open']) <= stops.get(s,-np.inf):
                        execute(s,0,min(float(bars[s].loc[ts,'open']),stops[s]),ts,'stop')
                        stopped.add(s)
                # Reductions apply immediately to existing positions once per scope.
                active = frozenset(risk.breaker.active)
                if mode not in ('buy_hold','sma200') and state == TradingState.REDUCED and active != getattr(risk,'_reduced_scope',None):
                    for s in shares:
                        execute(s,shares[s]//2,float(bars[s].loc[ts,'open']),ts,'risk_reduce')
                    risk._reduced_scope = active
                for signal in pending:
                    if not risk.can_trade():
                        break
                    s = signal.symbol
                    if s in stopped:
                        continue
                    price = float(bars[s].loc[ts,'open'])
                    equity = mark(ts,'open')
                    positions = {x:dict(market_value=shares[x]*float(bars[x].loc[ts,'open']),sector=sector[x]) for x in shares if shares[x]}
                    account = PortfolioState(equity,cash,max(0,equity*risk.config.max_leverage-sum(p['market_value'] for p in positions.values())),
                                             positions, timestamp=ts, flicker_rate=signal.metadata.get('flicker_rate',0),
                                             returns=returns.loc[:ts].iloc[:-1])
                    signal = replace(signal, entry_price=self.apply_slippage(price,'buy'))
                    desired = int(equity*signal.metadata['target_weight']/price)
                    if desired < shares[s]:
                        execute(s,desired,price,ts,'rebalance')
                    else:
                        decision = risk.validate_signal(signal,account)
                        if decision.approved:
                            execute(s,min(desired,decision.quantity),price,ts,'rebalance',dict(regime=signal.regime_name,confidence=signal.confidence))
                            stops[s] = signal.stop_loss
                pending = []
            if mode not in ('buy_hold','sma200'):
                for s in shares:
                    if shares[s] and float(bars[s].loc[ts,'low']) <= stops.get(s,-np.inf):
                        execute(s,0,min(float(bars[s].loc[ts,'open']),stops[s]),ts,'stop')
            # Closing marks also trigger breakers; liquidate next open (causal execution).
            equity = mark(ts,'close')
            risk._last_equity = equity
            risk.peak_equity = max(risk.peak_equity or equity,equity)
            closing_state = risk.evaluate_state(equity,[s for s in shares if shares[s]],last_regime)
            pending_liquidation = closing_state in (TradingState.HALTED,TradingState.KILLED)
            state_obj = None
            if mode == 'strategy':
                state_obj = engine.update(features.loc[ts],ts,log_events=False)
                if shuffle_labels:
                    rid = int(label_permutation[state_obj.state_id])
                    state_obj = replace(state_obj,state_id=rid,label=engine.regime_info[rid].regime_name)
                last_regime = state_obj.label
                signals = strategy.generate_signals(list(bars),{s:b.loc[:ts] for s,b in bars.items()},state_obj,engine.is_flickering())
            else:
                from core.regime_strategies import Signal, Direction
                signals=[]
                for s,b in bars.items():
                    price=float(b.loc[ts,'close'])
                    if mode == 'buy_hold':
                        allocation=1.
                    elif mode == 'sma200':
                        avg=b.loc[:ts,'close'].tail(200)
                        allocation=float(len(avg)==200 and price>avg.mean())
                    else:
                        allocation=float(rng.uniform(0,1))
                    # Buy-and-hold and SMA are conventional unlevered benchmarks;
                    # random allocations go through the same risk gate as the strategy.
                    signals.append(Signal(s,Direction.LONG,1.,price,price*.95,None,allocation,1.,0,mode,1.,ts,'benchmark',mode,
                                          {'target_weight':allocation/len(bars),'vol_regime':'mid'}))
                last_regime=mode
            current=sum(shares[s]*float(bars[s].loc[ts,'close']) for s in shares)/equity if equity>0 else 0
            target=sum(min(s.metadata['target_weight'],risk.config.max_single_position) for s in signals)
            # Compare the executable target with current allocation, not the uncapped 95% proposal.
            if strategy.needs_rebalance(current,target) and risk.can_trade():
                pending=signals
            if mode in ('buy_hold','sma200'):
                # These benchmarks intentionally retain their textbook allocation.
                pending=[]
                # Queue a target for next open using only today's trend decision.
                self_target={s.symbol:s.metadata['target_weight'] for s in signals}
                risk._benchmark_next=self_target
            records.append((ts,equity))
            weight_rows.append({s:shares[s]*float(bars[s].loc[ts,'close'])/equity for s in shares})
            history.append(dict(timestamp=ts,regime=last_regime,confidence=state_obj.probability if state_obj else 1.,
                                state_id=state_obj.state_id if state_obj else 0))
        eq=pd.Series(dict(records),name='equity',dtype=float)
        ret=eq.pct_change(fill_method=None)
        ret.iloc[0]=eq.iloc[0]/c.initial_capital-1
        return BacktestResult(eq,ret,pd.DataFrame(weight_rows,index=eq.index),
                              pd.Series([h['regime'] for h in history],index=eq.index,name='regime'),
                              pd.DataFrame(trades,columns=['timestamp','symbol','side','quantity','price','pnl','holding_days','regime','confidence','reason']),
                              folds,pd.DataFrame(history).set_index('timestamp'),c.initial_capital,risk.breaker.get_history())

    def run_fold(self, fold, bars_by_symbol, equity):
        clone=copy.copy(self)
        clone.config=replace(self.config,initial_capital=equity)
        bars={s:validate_bars(b) for s,b in bars_by_symbol.items()}
        features=self.features.build_features(next(iter(bars.values()))).loc[:fold.test_end]
        with TemporaryDirectory() as directory:
            risk=RiskManager(replace(self.risk.config,lock_path=str(Path(directory)/'halt.lock')))
            return clone._simulate(bars,features,[fold],risk,'strategy',0,False)

    def compare(self, bars_by_symbol, seeds=None):
        from backtest.performance import PerformanceReport
        results={mode:self.run(bars_by_symbol,mode=mode) for mode in ('buy_hold','sma200')}
        count=seeds if seeds is not None else self.config.random_seeds
        if count<1:
            raise ValueError('At least one random seed required')
        random=[PerformanceReport.from_result(self.run(bars_by_symbol,mode='random',seed=i),self.config.risk_free_rate).to_dict() for i in range(count)]
        table=pd.DataFrame(random).replace([np.inf,-np.inf],np.nan)
        return results, pd.DataFrame({'mean':table.mean(),'std':table.std(ddof=0)})
