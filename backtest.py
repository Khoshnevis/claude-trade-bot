#!/usr/bin/env python3
"""Backtest the live engines over MT5 history.

Usage:
    python backtest.py [config.yaml] [--bars N]

Pulls historical bars from the (already logged-in) MT5 terminal, replays the
exact KalmanPairsEngine / TrendEngine used live, and prints a metrics report.

A single backtest path is weak evidence. Treat a good result as "not yet
falsified", not as proof. Demo/live costs differ from the model.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from trade_bot.backtester import _BARS_PER_YEAR, Backtester, Result, Trade
from trade_bot.config import load_config
from trade_bot.logging_setup import setup_logging
from trade_bot.mt5_client import MT5Client, timeframe_const


def _meta_from_symbol_info(info) -> SimpleNamespace:
    return SimpleNamespace(
        name=info.name,
        trade_tick_value=info.trade_tick_value,
        trade_tick_size=info.trade_tick_size,
        point=info.point,
        digits=info.digits,
        volume_min=info.volume_min,
        volume_step=info.volume_step,
        volume_max=info.volume_max,
    )


def load_history(cfg, bars: int) -> tuple[dict, dict, dict]:
    client = MT5Client(terminal_path=cfg.mt5["terminal_path"])
    client.connect()
    tf = timeframe_const(cfg.engine["timeframe"])
    ctx_name = cfg.trend.get("context_timeframe") if cfg.trend.get("enabled") else None
    ctx_tf = timeframe_const(ctx_name) if ctx_name else None

    symbols = cfg.all_symbols()
    client.ensure_symbols(symbols)
    data, context, meta = {}, {}, {}
    for sym in symbols:
        data[sym] = client.rates(sym, tf, bars)
        meta[sym] = _meta_from_symbol_info(client.symbol_info(sym))
    if ctx_tf is not None:
        for sym in cfg.trend.get("symbols", []):
            context[sym] = client.rates(sym, ctx_tf, bars)
    client.shutdown()
    return data, context, meta


def _sub_result(res: Result, lo_pos: int, hi_pos: int) -> Result:
    """Slice a Result by equity-curve position range [lo_pos, hi_pos)."""
    curve = res.equity_curve.iloc[lo_pos:hi_pos]
    lo_i, hi_i = res.warmup + lo_pos, res.warmup + hi_pos
    trades = [t for t in res.trades if lo_i <= t.close_i < hi_i]
    start = float(curve.iloc[0]) if len(curve) else res.start_equity
    end = float(curve.iloc[-1]) if len(curve) else start
    return Result(start, end, curve, trades, {}, res.warmup)


def print_oos(cfg, res: Result) -> None:
    """Split the run in half and compare in-sample vs out-of-sample.

    If an edge is real it should persist into the unseen second half. A result
    that only looks good in-sample is overfit. This is the honest test.
    """
    n = len(res.equity_curve)
    if n < 20:
        return
    mid = n // 2
    is_res = _sub_result(res, 0, mid)
    oos_res = _sub_result(res, mid, n)
    bpy = _BARS_PER_YEAR.get(cfg.engine["timeframe"], 252)
    print("\n" + "=" * 56)
    print("  IN-SAMPLE vs OUT-OF-SAMPLE (honest consistency check)")
    print("=" * 56)
    print(f"  {'metric':<18}{'in-sample':>16}{'out-of-sample':>18}")
    im, om = is_res.metrics(bpy), oos_res.metrics(bpy)
    for key, label, scale, suf in [
        ("total_return", "return", 100, "%"),
        ("sharpe_annual", "sharpe", 1, ""),
        ("max_drawdown", "max drawdown", 100, "%"),
        ("n_trades", "trades", 1, ""),
        ("win_rate", "win rate", 100, "%"),
        ("expectancy", "expectancy", 1, ""),
    ]:
        iv, ov = im[key] * scale, om[key] * scale
        if key in ("n_trades",):
            print(f"  {label:<18}{iv:>16.0f}{ov:>18.0f}")
        else:
            print(f"  {label:<18}{iv:>15.2f}{suf}{ov:>17.2f}{suf}")
    print("=" * 56)
    print("  READ THIS: if out-of-sample is far worse than in-sample, the")
    print("  result is overfit. Only an edge that survives OOS is real.")
    print("=" * 56 + "\n")


def print_report(cfg, res: Result) -> None:
    tf = cfg.engine["timeframe"]
    bpy = _BARS_PER_YEAR.get(tf, 252)
    m = res.metrics(bpy)
    print("\n" + "=" * 56)
    print(f"  BACKTEST REPORT  ({tf}, {res.n_trades} trades)")
    print("=" * 56)
    print(f"  Start equity     : {res.start_equity:,.2f}")
    print(f"  End equity       : {m['end_equity']:,.2f}")
    print(f"  Total return     : {m['total_return']*100:+.2f}%")
    print(f"  Annualized Sharpe: {m['sharpe_annual']:.2f}")
    print(f"  Max drawdown     : {m['max_drawdown']*100:.2f}%")
    print(f"  Trades           : {m['n_trades']}")
    print(f"  Win rate         : {m['win_rate']*100:.1f}%")
    print(f"  Avg win / loss   : {m['avg_win']:,.2f} / {m['avg_loss']:,.2f}")
    print(f"  Expectancy/trade : {m['expectancy']:+,.2f}")
    pf = m["profit_factor"]
    print(f"  Profit factor    : {'inf' if pf == float('inf') else f'{pf:.2f}'}")
    print("=" * 56)
    if res.diag:
        print("  WHY-NO-TRADE DIAGNOSTICS (bar-evaluation tallies)")
        labels = {
            "pair_coint_block": "pairs: cointegration gate blocked",
            "pair_cointegrated_no_entry": "pairs: cointegrated but |z| < entry",
            "pair_entry_signal": "pairs: entry signals fired",
            "pair_blocked_by_risk": "pairs: blocked by risk gate",
            "pair_size_skip": "pairs: skipped (lot < volume_min)",
            "pair_opened": "pairs: positions opened",
            "trend_regime_block": "trend: HMM regime = ranging (blocked)",
            "trend_no_setup": "trend: no MTF+breakout setup",
            "trend_entry_signal": "trend: entry signals fired",
            "trend_blocked_by_risk": "trend: blocked by risk gate",
            "trend_size_skip": "trend: skipped (lot < volume_min)",
            "trend_opened": "trend: positions opened",
            "reversal_no_setup": "reversal: no top/bottom setup",
            "reversal_entry_signal": "reversal: entry signals fired",
            "reversal_blocked_by_risk": "reversal: blocked by risk gate",
            "reversal_size_skip": "reversal: skipped (lot < volume_min)",
            "reversal_opened": "reversal: positions opened",
        }
        for k, label in labels.items():
            if k in res.diag:
                print(f"    {label:<42}: {res.diag[k]}")
        print("=" * 56)
    print("  NOTE: single-path backtest = weak evidence. Costs are modeled, not")
    print("  guaranteed. Validate out-of-sample before risking real capital.")
    print("=" * 56 + "\n")


def main() -> int:
    args = [a for a in sys.argv[1:]]
    show_oos = "--oos" in args
    if show_oos:
        args.remove("--oos")
    bars = None
    if "--bars" in args:
        idx = args.index("--bars")
        bars = int(args[idx + 1])
        del args[idx:idx + 2]
    cfg_path = args[0] if args else "config.yaml"
    cfg = load_config(cfg_path)
    setup_logging(cfg["logging"]["level"], None)
    bars = bars or cfg["backtest"]["bars"]

    print(f"Loading {bars} bars of history from MT5...")
    data, context, meta = load_history(cfg, bars)
    bpy = _BARS_PER_YEAR.get(cfg.engine["timeframe"], 252)
    bt = Backtester(cfg, data, context, meta, bpy)
    res = bt.run()
    print_report(cfg, res)
    if show_oos:
        print_oos(cfg, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
