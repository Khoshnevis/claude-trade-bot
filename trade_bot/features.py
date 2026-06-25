"""Pure, look-ahead-safe feature functions shared by all engines.

Every function consumes only data up to and including the current bar, so the
same code is correct whether called in research or live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR over the last `period` bars, in price units."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def sma(series: pd.Series, window: int) -> float:
    return float(series.rolling(window).mean().iloc[-1])


def donchian(df: pd.DataFrame, window: int) -> tuple[float, float]:
    """Upper/lower Donchian channel computed on bars BEFORE the current one."""
    upper = float(df["high"].iloc[-window - 1:-1].max())
    lower = float(df["low"].iloc[-window - 1:-1].min())
    return upper, lower


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).dropna()


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return log_returns(series).rolling(window).std()
