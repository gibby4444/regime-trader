Step 7 runs through `main.py live` (or no subcommand), with `--dry-run` exercising
features, forward filtering, strategy, risk, stops, breakers and dashboard without
broker writes. `train-only`/`--train-only`, `backtest`/`--backtest`, `--stress-test`
and `--compare` dispatch to their respective pipelines. Construction/imports do
not connect. The existing Alpaca adapter connects only on explicit execution.

Live bars default to `broker.live_timeframe: 5Min`; training uses completed daily
bars. The initial feature window comes from completed historical live bars. New
observations are processed once per completed universe timestamp, on the owner
thread. `TradingLoop` accepts broker, stream, clock, sleep, training, historical
warmup, monitoring and file-path dependencies for offline testing.

Snapshots are atomic JSON files. Recovery preserves risk baselines, calendar
scopes, breaker state and position metadata. Current broker quantities and stops
remain authoritative. Forward/stability state is recovered only for the same
model training timestamp. Retraining clears the previous model's filter state.
Feed failure pauses entries and inference; existing broker stops remain active,
and new bars can tighten them. `feed_recovered()` explicitly resumes inference
after the stream owner confirms reconnection. Shutdown signals only request a
stop; cleanup closes streams and saves a snapshot without order/cancel calls.

Clarifications/deviations from the terse course text:

- Breakers also run before signals, so a newly observed loss or lock blocks entries
  on that bar. Breaker exits use Step 5's entry-veto bypass.
- API **reads/connect** use three total attempts, with injectable 1s/2s exponential
  delays. Ambiguous order submissions are **never replayed**, preserving Step 5's
  existing timeout safety and tests; broker acceptance may precede a timeout.
- Market-closed operation reports the broker's next open and exits (the course
  permits wait or exit). No approximate holiday calendar is invented.
- Alerts default to local critical log messages. External email/webhook delivery
  is not configured. The formerly stubbed terminal dashboard is implemented.
- Entry orders include GTC protective OTO stops. Approved quantities represent
  target holdings; submissions use the difference from existing holdings.

All verification uses fake Alpaca clients/SDKs, streams, wall clocks and sleepers.
No live/paper loop was executed, no credentials were read, and no orders were sent.
