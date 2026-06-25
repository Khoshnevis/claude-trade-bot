"""Single-instrument swing-reversal / mean-reversion engine.

Idea (matches the "detect a top/bottom, confirm with filters, then enter"
workflow): a swing low/high must form, AND price must be stretched from its
moving average by several ATR, AND RSI must be oversold/overbought. Only when
these independent-ish filters agree do we take a counter-move trade.

Exits are MEAN-REVERSION based (price returns to the mean / RSI normalizes).
The hard stop/target are SOFT -- the engine stores them and monitors price
itself rather than parking them on the broker, which allows intrabar tick
monitoring and flexible exits.

This engine naturally fires more often than breakout trend-following and suits
FX ranging behaviour better than fragile intraday cointegration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..features import atr, is_swing_high, is_swing_low, rsi, sma

log = logging.getLogger("trade_bot.reversal")


@dataclass
class ReversalSignal:
    symbol: str
    action: str          # 'enter_long' | 'enter_short' | 'hold'
    atr: float
    rsi: float
    note: str = ""


class ReversalEngine:
    def __init__(self, cfg: dict, atr_period: int = 14):
        self.cfg = cfg
        self.rsi_period = cfg.get("rsi_period", 14)
        self.rsi_os = cfg.get("rsi_oversold", 30)
        self.rsi_ob = cfg.get("rsi_overbought", 70)
        self.ma_period = cfg.get("ma_period", 50)
        self.stretch_atr = cfg.get("stretch_atr", 1.0)
        self.pivot_lookback = cfg.get("pivot_lookback", 20)
        self.pivot_confirm = cfg.get("pivot_confirm", 2)
        self.atr_period = atr_period

    def _snapshot(self, df: pd.DataFrame) -> dict:
        close = df["close"]
        return {
            "price": float(close.iloc[-1]),
            "ma": sma(close, self.ma_period),
            "atr": atr(df, self.atr_period),
            "rsi": rsi(close, self.rsi_period),
        }

    def evaluate(self, symbol: str, df: pd.DataFrame) -> ReversalSignal | None:
        need = max(self.ma_period, self.pivot_lookback, self.atr_period,
                   self.rsi_period) + 5
        if len(df) < need:
            return None
        s = self._snapshot(df)
        if s["atr"] <= 0:
            return None
        stretch = self.stretch_atr * s["atr"]

        long_filters = (
            s["rsi"] <= self.rsi_os
            and (s["ma"] - s["price"]) >= stretch
            and is_swing_low(df, self.pivot_lookback, self.pivot_confirm)
        )
        short_filters = (
            s["rsi"] >= self.rsi_ob
            and (s["price"] - s["ma"]) >= stretch
            and is_swing_high(df, self.pivot_lookback, self.pivot_confirm)
        )
        if long_filters:
            return ReversalSignal(symbol, "enter_long", s["atr"], s["rsi"],
                                  note="bottom + oversold + stretched")
        if short_filters:
            return ReversalSignal(symbol, "enter_short", s["atr"], s["rsi"],
                                  note="top + overbought + stretched")
        return ReversalSignal(symbol, "hold", s["atr"], s["rsi"])

    def reversion_done(self, df: pd.DataFrame, side: int) -> bool:
        """Soft take-profit: True once price has reverted to the mean / RSI
        has normalized. `side` is +1 long, -1 short."""
        s = self._snapshot(df)
        if side > 0:
            return s["price"] >= s["ma"] or s["rsi"] >= 50
        return s["price"] <= s["ma"] or s["rsi"] <= 50
