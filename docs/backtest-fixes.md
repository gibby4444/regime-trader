# Backtest execution corrections (2026-10-19)

Course risk constraints remain unchanged: 1% trade risk, 80% gross exposure,
15% single position and 30% sector exposure. The local verbatim course prompt
(`/workspace/cursor-build/course-notes/track-b-L1-L8-prompts-verbatim.md`, lines
545–574) gives the risk manager absolute veto and caps every position at 15%.
It provides no single-symbol exemption. Earlier strategy lessons propose
60–95% allocations, but those proposals do not override subsequent risk limits.
Consequently a SPY-only account remains approximately 15% invested; the 80%
portfolio cap is not the binding constraint. We do not invent an exemption.

The EMA-based course stop formula can put a long stop above market in a falling
market. Reject such entry/add signals instead of changing the formula or filling
an immediately triggered bracket. The risk manager now requires stop < entry;
the simulator additionally checks stop < the next executable open (before
slippage), to handle overnight gaps. This matches the broker executor's existing
long-stop validation. Reductions remain executable regardless of the proposed
new stop. Position sizing also returns zero for invalid long stop geometry.

Signals generated at close still fill next open with 0.05% per-side slippage.
A valid standing stop remains active on the entry day: low <= stop legitimately
exits at stop, or at the open for an existing position gapping through its stop.
Skipping all entry-day stops would conceal real losses. The broker's limit-price
0.1% offset is not part of this daily next-open simulation.

A pending rebalance used to overwrite its stop even when the risk-approved
quantity produced no buy. Standing stops now update only on actual entry/add
fills; adds may tighten but never widen the stop. Stops are not recomputed on
every bar. Export weights.csv alongside account equity for exposure auditing.

Regression coverage includes stops above/equal to entry, next-open gaps,
stops between raw open and slipped fill, legitimate entry-day stops, unaffected
distant stops, no-fill stop replacement, and stop widening on an add. No risk,
strategy, training, or rebalance parameters were tuned to improve performance.
