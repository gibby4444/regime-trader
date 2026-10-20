"""Performance statistics for account equity and completed (sell-side) trades."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
import numpy as np
import pandas as pd
TRADING_DAYS=252

def sharpe_ratio(returns, risk_free_rate=.045):
    r=returns.dropna()-risk_free_rate/TRADING_DAYS
    std=r.std(ddof=0)
    return float(np.sqrt(TRADING_DAYS)*r.mean()/std) if len(r) and std>1e-15 else 0.

def sortino_ratio(returns, risk_free_rate=.045):
    r=returns.dropna()-risk_free_rate/TRADING_DAYS
    downside=float(np.sqrt(np.minimum(r,0).pow(2).mean())) if len(r) else 0
    return float(np.sqrt(TRADING_DAYS)*r.mean()/downside) if downside>0 else 0.

def cagr(equity):
    if len(equity)<2 or equity.iloc[0]<=0:
        return 0.
    return float((max(0,equity.iloc[-1])/equity.iloc[0])**(TRADING_DAYS/(len(equity)-1))-1)

def drawdown_series(equity):
    return equity/equity.cummax()-1

def max_drawdown(equity):
    return float(-drawdown_series(equity).min()) if len(equity) else 0.

def calmar_ratio(equity):
    dd=max_drawdown(equity)
    return cagr(equity)/dd if dd else 0.

def underwater_duration(equity):
    longest=current=0
    for dd in drawdown_series(equity):
        current=current+1 if dd<0 else 0
        longest=max(longest,current)
    return longest

def regime_breakdown(returns, regimes, trades=None):
    frame=pd.DataFrame({'return':returns,'regime':regimes}).dropna()
    frame['contribution']=frame['return']*(1+frame['return']).cumprod().shift(1,fill_value=1.)
    rows=[]
    for regime,g in frame.groupby('regime'):
        t=trades[(trades.regime==regime)&(trades.side=='sell')] if trades is not None and not trades.empty else pd.DataFrame()
        rows.append(dict(regime=regime,bars=len(g),time_pct=len(g)/len(frame),return_contribution=g['contribution'].sum(),
                         avg_trade_pnl=float(t.pnl.mean()) if len(t) else 0.,
                         win_rate=float((t.pnl>0).mean()) if len(t) else 0.,sharpe=sharpe_ratio(g['return'])))
    return pd.DataFrame(rows).set_index('regime') if rows else pd.DataFrame()

def confidence_breakdown(trades):
    rows=[]
    for label,lo,hi in [('<50%',0,.5),('50-60%',.5,.6),('60-70%',.6,.7),('70%+',.7,1.000001)]:
        t=trades[(trades.side=='sell')&(trades.confidence>=lo)&(trades.confidence<hi)] if not trades.empty else trades
        # Trade-return Sharpe uses realised return on exit notional (not dollar P&L).
        ret=t.pnl/(t.quantity*t.price) if len(t) else pd.Series(dtype=float)
        rows.append(dict(confidence=label,trades=len(t),sharpe=sharpe_ratio(ret,0),
                         win_rate=float((t.pnl>0).mean()) if len(t) else 0.,avg_pnl=float(t.pnl.mean()) if len(t) else 0.))
    return pd.DataFrame(rows).set_index('confidence')

def benchmark_comparison(returns, benchmark_returns, risk_free_rate=.045):
    f=pd.concat([returns,benchmark_returns],axis=1).dropna()
    if len(f)<2:
        return dict(alpha=0.,beta=0.,correlation=0.,tracking_error=0.,information_ratio=0.)
    r,b=f.iloc[:,0],f.iloc[:,1]
    beta=float(r.cov(b)/b.var()) if b.var()>0 else 0.
    active=r-b
    tracking=float(active.std()*np.sqrt(252))
    return dict(alpha=float((r.mean()-risk_free_rate/252-beta*(b.mean()-risk_free_rate/252))*252),beta=beta,
                correlation=float(r.corr(b)) if r.std()>0 and b.std()>0 else 0.,tracking_error=tracking,
                information_ratio=float(active.mean()*252/tracking) if tracking else 0.)

@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    regime_stats: pd.DataFrame
    benchmark: dict
    monthly_returns: pd.DataFrame | None = None
    max_drawdown_duration: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    total_trades: int = 0
    avg_holding_period: float = 0
    worst_day: float = 0
    worst_week: float = 0
    worst_month: float = 0
    max_consecutive_losses: int = 0
    longest_underwater: int = 0
    confidence_stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_backtest(cls,equity,regimes,benchmark_equity=None,risk_free_rate=.045,trades=None,initial_capital=None):
        if equity.empty:
            raise ValueError('Empty equity curve')
        initial=initial_capital if initial_capital is not None else float(equity.iloc[0])
        # Include starting capital so first OOS loss/slippage counts in every metric.
        start=equity.index[0]-pd.offsets.BDay(1)
        curve=pd.concat([pd.Series([initial],index=[start]),equity])
        r=curve.pct_change(fill_method=None).iloc[1:]
        total=float(equity.iloc[-1]/initial-1)
        annual=float((max(0,equity.iloc[-1])/initial)**(252/len(equity))-1)
        dd=max_drawdown(curve)
        weekly=(1+r).resample('W').prod()-1
        monthly=(1+r).resample('ME').prod()-1
        t=trades if trades is not None else pd.DataFrame()
        closed=t[t.side=='sell'] if len(t) else t
        wins=closed.pnl[closed.pnl>0].sum() if len(closed) else 0
        losses=-closed.pnl[closed.pnl<0].sum() if len(closed) else 0
        consecutive=current=0
        for p in closed.pnl if len(closed) else []:
            current=current+1 if p<0 else 0
            consecutive=max(consecutive,current)
        duration=underwater_duration(curve)
        return cls(total,annual,sharpe_ratio(r,risk_free_rate),sortino_ratio(r,risk_free_rate),dd,
                   annual/dd if dd else 0.,regime_breakdown(r,regimes,t),
                   benchmark_comparison(r,benchmark_equity.pct_change(),risk_free_rate) if benchmark_equity is not None else {},
                   monthly.to_frame('return'),duration,float((closed.pnl>0).mean()) if len(closed) else 0.,
                   float(wins/losses) if losses else (float('inf') if wins else 0.),len(t),
                   float(closed.holding_days.mean()) if len(closed) else 0.,float(r.min()),float(weekly.min()),float(monthly.min()),
                   consecutive,duration,confidence_breakdown(t))

    @classmethod
    def from_result(cls,result,risk_free_rate=.045):
        return cls.from_backtest(result.equity_curve,result.regimes,risk_free_rate=risk_free_rate,
                                 trades=result.trades,initial_capital=result.initial_capital)

    def to_dict(self):
        return {f.name:getattr(self,f.name) for f in fields(self) if isinstance(getattr(self,f.name),(int,float,np.floating))}

    def print_summary(self):
        from rich.console import Console
        from rich.table import Table
        console=Console()
        for name,frame in [('Performance',pd.Series(self.to_dict(),name='value').to_frame()),('Regimes',self.regime_stats),('Confidence',self.confidence_stats)]:
            table=Table(title=name)
            table.add_column('Metric')
            for col in frame:
                table.add_column(str(col))
            for idx,row in frame.iterrows():
                table.add_row(str(idx),*[f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row])
            console.print(table)
