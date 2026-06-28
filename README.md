# claude-trade-bot

A fully automated, low-frequency algorithmic forex trading bot for **MetaTrader 5**,
built for a small (~$8k) EU/MiFID account where transaction cost is the binding
constraint. It runs two uncorrelated, slow (H1/H4) engines and uses online/adaptive
estimators to *filter and size* signals rather than predict prices from scratch.

> The MT5 terminal is assumed to be **already running and logged in**. The bot
> attaches over the local IPC bridge **without credentials** (`mt5.initialize()`).

## Why this design

At ~$8k the dominant constraint is cost drag (~0.6–0.8 pip round-turn on a raw/ECN
account), so only **low-turnover** strategies survive. Scalping and HFT are not
viable for retail at this size, and the Python↔terminal IPC bridge adds latency that
rules out sub-second trading anyway. Hence: slow bars, market-neutral core, and
strict risk control.

## The two engines

1. **Kalman / cointegration pairs** (`trade_bot/signals/kalman_pairs.py`) — the
   market-neutral core. A hand-rolled 2-state Kalman filter estimates a *time-varying*
   hedge ratio for an economically-linked pair (e.g. EURUSD/GBPUSD), forms the spread,
   and trades reversion on the normalized forecast error (z-score). An optional
   Engle-Granger cointegration gate (`statsmodels`) refuses pairs whose relationship
   has decayed. No arbitrary rolling window; adapts through structural breaks.

2. **Regime-filtered trend overlay with multi-timeframe confirmation**
   (`trade_bot/signals/trend.py` + `signals/regime.py`) — direction and the
   Gaussian-HMM regime gate are read from a higher *context* timeframe
   (`trend.context_timeframe`, e.g. H4), while the Donchian breakout entry and
   ATR stop fire on the base timeframe (e.g. H1). A fast-TF breakout is only
   taken when it agrees with the slow-TF trend and the regime is "trending".
   This raises signal quality rather than just increasing trade frequency.

3. **Single-instrument swing-reversal / mean-reversion** (`trade_bot/signals/reversal.py`)
   — the higher-frequency engine. Detects a swing top/bottom, confirms with RSI
   (oversold/overbought) **and** price being stretched several ATR from its mean,
   then enters a counter-move sized by risk. Exits are mean-reversion based
   (price returns to the mean / RSI normalizes). Its **SL/TP are SOFT**: the bot
   stores them and monitors price on every poll (intrabar), closing the position
   itself rather than parking stops on the broker. Several concurrent positions
   across symbols are allowed.

Both `statsmodels` and `hmmlearn` are **optional**: if missing, the bot logs a warning
and degrades gracefully (cointegration gate off / regime always-on).

### Gold (XAUUSD) meta-labeling ML

`trade_bot/ml/` is a meta-labeling pipeline (López de Prado style). The ML model
does **not** predict price. A rule-based *primary* signal (a breakout that agrees
with the EMA trend — gold trends well) proposes a side+timing; a scikit-learn
classifier estimates `P(this signal is profitable)` from features (RSI, ADX,
ATR, distance-from-EMA, volatility, hour, …), and only signals above a
probability threshold are taken. Training labels come from the **triple-barrier**
method, and evaluation is strictly **out-of-sample** (train on the first 70%,
test on the unseen last 30%).

```bash
python train_gold_ml.py --symbol XAUUSD --tf H1 --bars 6000 --threshold 0.55
```

The report compares OOS economics of *taking all primary signals* vs *taking
only ML-approved ones*. The ML layer earns its place **only if it raises
out-of-sample expectancy**; a large train-vs-OOS AUC gap means overfitting and
the model should not be trusted. Models are saved to `models/` (gitignored).

## Risk management (`trade_bot/risk.py`)

- **Fixed-fractional sizing:** 0.5% of equity risked per trade (hard ceiling 1%),
  sized from the symbol's tick value/size so it is correct for any quote currency.
- **ATR-based stops** for trend trades; **z-score blow-out stop** for pairs.
- **Portfolio heat cap** (2%), **max open positions**, and **correlation limits**
  (no stacking same-direction trades within a correlated cluster).
- **Kill-switch:** flatten everything and halt new trades at a 12% drawdown from the
  high-water mark; a separate 3% daily-loss halt (cleared at the new trading day).

## Architecture (clean separation of concerns)

```
run.py                      entry point
trade_bot/
  config.py                 YAML config + validation
  mt5_client.py             resilient MT5 IPC wrapper (reconnect w/ backoff)
  features.py               pure, look-ahead-safe indicators (ATR, MA, Donchian)
  signals/
    kalman_pairs.py         Kalman cointegration pairs engine
    trend.py                trend/momentum overlay
    regime.py               Gaussian-HMM regime gate
  risk.py                   sizing, portfolio limits, kill-switch (veto power)
  execution.py              OMS: order_send, comment-tag ownership, pair legs
  persistence.py            SQLite state (high-water, daily anchor, audit trail)
  engine.py                 orchestration: act once per closed bar; reconcile
```

Position ownership is tracked by the bot's **magic number** (isolates it from humans
and other EAs) plus **comment tags** (`TR|SYMBOL`, `KP|AB|y`/`|x`), so a restart
re-derives state purely from the broker — local state is never authoritative.

## Setup (Windows VPS, co-located near the broker)

1. Install and **log into** the MT5 terminal; enable **Algo Trading** (the toolbar
   button must be green).
2. `pip install -r requirements.txt`
3. Edit `config.yaml` — set your symbols/pairs (use your broker's exact symbol names,
   e.g. `EURUSD` vs `EURUSD.r`), risk, and timeframe.
4. `python run.py`

For 24/5 operation: auto-login the Windows session (Sysinternals Autologon), launch
the terminal from the Startup folder, and disconnect RDP with `tscon` (not log off) so
the GUI terminal keeps running.

## Configuration

All behavior is driven by `config.yaml` (see inline comments). Key knobs: `engine.timeframe`,
`pairs.entry_z` / `exit_z` / `stop_z`, `trend.fast_ma` / `slow_ma` / `donchian`, and the
entire `risk` block. The bot only trades symbols referenced under `pairs` or `trend`.

## Operational notes & honest expectations

- This bot **places real orders**. Start on a **demo account** and run it 24/5 for
  several months before risking real capital, then go live tiny (0.25–0.5% risk).
- The realistic goal at this capital is to prove a small, robust, positive-expectancy
  edge and operational competence — **not** to generate a living. Returns promising
  5–10%/month are a red flag. Most automated FX attempts fail; rigorous risk control
  and low turnover are what keep you in the surviving minority.
- The bot is appropriate for **minute-to-hour cadence only**, never sub-second.

## License

Private. Use at your own risk — trading involves substantial risk of loss.
