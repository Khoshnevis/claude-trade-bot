"""Slow trend / momentum overlay with multi-timeframe confirmation.

Multi-timeframe (MTF) design:
  * DIRECTION + REGIME come from a higher "context" timeframe (e.g. H4/D1):
    the dual-MA alignment sets the only allowed trade direction, and the HMM
    regime gate (computed on that higher TF, where it is more stable) must say
    "trending".
  * ENTRY TRIGGER + RISK (ATR stop) come from the base timeframe (e.g. H1):
    a Donchian breakout in the allowed direction fires the entry.

This raises signal quality (a fast-TF breakout is only taken when it agrees
with the slow-TF trend), rather than just increasing trade frequency. If no
context frame is supplied it falls back to single-timeframe behaviour.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..features import atr, donchian, sma
from .regime import RegimeFilter

log = logging.getLogger("trade_bot.trend")


@dataclass
class TrendSignal:
    symbol: str
    action: str          # 'enter_long' | 'enter_short' | 'exit_*_if_held' | 'hold'
    atr: float
    note: str = ""


class TrendEngine:
    def __init__(self, cfg: dict, atr_period: int = 14):
        self.cfg = cfg
        self.fast = cfg["fast_ma"]
        self.slow = cfg["slow_ma"]
        self.donchian = cfg["donchian"]
        self.atr_period = atr_period
        self.use_regime = cfg.get("use_hmm_regime", False)
        self.regime = (
            RegimeFilter(
                n_states=cfg.get("hmm_states", 3),
                retrain_bars=cfg.get("hmm_retrain_bars", 500),
                lookback=cfg.get("hmm_lookback", 1000),
            )
            if self.use_regime
            else None
        )

    def evaluate(self, symbol: str, df: pd.DataFrame,
                 context_df: pd.DataFrame | None = None) -> TrendSignal | None:
        """`df` is the base (entry) timeframe; `context_df` is the higher TF.

        If `context_df` is None the higher-TF checks fall back to `df`.
        """
        need = max(self.donchian, self.atr_period) + 5
        if len(df) < need:
            return None
        ctx = context_df if context_df is not None else df
        if len(ctx) < self.slow + 5:
            return None

        # --- Direction & regime from the higher (context) timeframe ---------
        ctx_close = ctx["close"]
        ctx_fast = sma(ctx_close, self.fast)
        ctx_slow = sma(ctx_close, self.slow)
        ctx_uptrend = ctx_fast > ctx_slow
        ctx_downtrend = ctx_fast < ctx_slow

        # --- Entry trigger & risk from the base timeframe -------------------
        price = float(df["close"].iloc[-1])
        upper, lower = donchian(df, self.donchian)
        a = atr(df, self.atr_period)

        long_ok = ctx_uptrend and price >= upper
        short_ok = ctx_downtrend and price <= lower

        # Regime gate evaluated on the context timeframe (more stable there).
        if self.regime is not None and (long_ok or short_ok):
            if not self.regime.is_trending(ctx):
                return TrendSignal(symbol, "hold", a, note="regime=ranging")

        if long_ok:
            return TrendSignal(symbol, "enter_long", a, note="MTF up + H1 breakout")
        if short_ok:
            return TrendSignal(symbol, "enter_short", a, note="MTF down + H1 breakout")

        # Soft exit when the higher-TF trend flips against a held position.
        if ctx_downtrend:
            return TrendSignal(symbol, "exit_long_if_held", a)
        if ctx_uptrend:
            return TrendSignal(symbol, "exit_short_if_held", a)
        return TrendSignal(symbol, "hold", a)
