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

from trade_bot.backtester import _BARS_PER_YEAR, Backtester, Result
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
    print("  NOTE: single-path backtest = weak evidence. Costs are modeled, not")
    print("  guaranteed. Validate out-of-sample before risking real capital.")
    print("=" * 56 + "\n")


def main() -> int:
    args = [a for a in sys.argv[1:]]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
