"""Slow trend / momentum overlay with an HMM regime gate.

Long-only-direction breakout: a dual moving-average alignment must agree with a
Donchian channel breakout, and the HMM must classify the current regime as
trending. Trades H4/D1-style slow signals; one directional position per symbol.
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
    action: str          # 'enter_long' | 'enter_short' | 'exit' | 'hold'
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

    def evaluate(self, symbol: str, df: pd.DataFrame) -> TrendSignal | None:
        need = max(self.slow, self.donchian, self.atr_period) + 5
        if len(df) < need:
            return None

        close = df["close"]
        fast_ma = sma(close, self.fast)
        slow_ma = sma(close, self.slow)
        upper, lower = donchian(df, self.donchian)
        price = float(close.iloc[-1])
        a = atr(df, self.atr_period)

        long_ok = fast_ma > slow_ma and price >= upper
        short_ok = fast_ma < slow_ma and price <= lower

        if self.regime is not None and (long_ok or short_ok):
            if not self.regime.is_trending(df):
                return TrendSignal(symbol, "hold", a, note="regime=ranging")

        if long_ok:
            return TrendSignal(symbol, "enter_long", a)
        if short_ok:
            return TrendSignal(symbol, "enter_short", a)

        # Exit signal: MA cross back against the position is handled by the
        # engine via stops, but we expose a soft exit when MAs flip.
        if fast_ma < slow_ma:
            return TrendSignal(symbol, "exit_long_if_held", a)
        if fast_ma > slow_ma:
            return TrendSignal(symbol, "exit_short_if_held", a)
        return TrendSignal(symbol, "hold", a)
