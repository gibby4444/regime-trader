"""Reproducible offline crash, overnight gap, and regime misclassification tests."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from data.feature_engineering import atr

@dataclass(frozen=True)
class StressScenario:
    name: str
    shock_pct: float
    duration_bars: int = 1
    start: pd.Timestamp | None = None
    gap: bool = False

class StressTester:
    def __init__(self,backtester,seed=None):
        self.backtester=backtester
        self.rng=np.random.default_rng(seed)

    def inject_crash(self,bars,shock_pct,start,duration_bars=1):
        if not -1<shock_pct<0 or duration_bars<1:
            raise ValueError('Crash must be between -100% and 0%; duration positive')
        out=bars.copy()
        loc=out.index.get_loc(start)
        factors=np.ones(len(out))
        n=min(duration_bars,len(out)-loc)
        factors[loc:loc+n]=(1+shock_pct)**(np.arange(1,n+1)/duration_bars)
        factors[loc+n:]=factors[loc+n-1]
        out.loc[:,['open','high','low','close']]=out[['open','high','low','close']].mul(factors,axis=0)
        return out

    def simulate_gap(self,bars,gap_pct,at):
        # Permanent repricing preserves valid OHLC and makes the opening gap real.
        return self.inject_crash(bars,gap_pct,at,1)

    def default_scenarios(self):
        return [StressScenario('crash_10pct',-.10),StressScenario('crash_15pct',-.15),StressScenario('gap_7pct',-.07,gap=True)]

    def run_scenario(self,scenario,bars_by_symbol):
        altered={}
        for s,b in bars_by_symbol.items():
            at=scenario.start or b.index[-max(2,len(b)//5)]
            altered[s]=(self.simulate_gap(b,scenario.shock_pct,at) if scenario.gap else
                        self.inject_crash(b,scenario.shock_pct,at,scenario.duration_bars))
        return self.backtester.run(altered)

    def run_all(self,bars_by_symbol,simulations=None,points=None):
        from backtest.performance import PerformanceReport
        sims=simulations if simulations is not None else self.backtester.config.stress_simulations
        points=points if points is not None else self.backtester.config.stress_points
        if sims<1 or points<1:
            raise ValueError('Simulation and shock counts must be positive')
        # Only shock OOS observations, so tests exercise an already trained account.
        b=next(iter(bars_by_symbol.values()))
        valid=self.backtester.features.build_features(b).index[self.backtester.config.train_window:]
        if not len(valid):
            raise ValueError('Insufficient OOS data')
        rows=[]
        for trial in range(sims):
            dates=self.rng.choice(valid,size=min(points,len(valid)),replace=False)
            crash={s:x.copy() for s,x in bars_by_symbol.items()}
            gaps={s:x.copy() for s,x in bars_by_symbol.items()}
            for date in sorted(dates):
                shock=-float(self.rng.uniform(.05,.15))
                multiple=float(self.rng.uniform(2,5))
                for s,original in bars_by_symbol.items():
                    crash[s]=self.inject_crash(crash[s],shock,pd.Timestamp(date),1)
                    history=gaps[s].loc[:date].iloc[:-1]
                    a=atr(history.high,history.low,history.close).iloc[-1]
                    gap=-min(.99,multiple*float(a)/float(history.close.iloc[-1]))
                    gaps[s]=self.simulate_gap(gaps[s],gap,pd.Timestamp(date))
            for name,data,shuffle in [('crash',crash,False),('overnight_gap',gaps,False),('regime_shuffle',bars_by_symbol,True)]:
                result=self.backtester.run(data,shuffle_labels=shuffle,seed=trial)
                rows.append(dict(scenario=name,simulation=trial,**PerformanceReport.from_result(result).to_dict(),
                                 breaker_triggers=len(result.breaker_history)))
        return pd.DataFrame(rows)
