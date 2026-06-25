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


def rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder's RSI of the last bar (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    last_loss = float(avg_loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = float(avg_gain.iloc[-1]) / last_loss
    return 100.0 - 100.0 / (1.0 + rs)


def is_swing_low(df: pd.DataFrame, lookback: int, confirm: int = 1) -> bool:
    """True if a swing low formed recently: the lowest low of the lookback
    window is within the last `confirm` bars and price is turning back up."""
    lows = df["low"]
    window = lows.iloc[-lookback:]
    recent_min_pos = window.values.argmin()
    bars_since = len(window) - 1 - recent_min_pos
    turning_up = float(df["close"].iloc[-1]) > float(df["close"].iloc[-2])
    return bars_since <= confirm and turning_up


def is_swing_high(df: pd.DataFrame, lookback: int, confirm: int = 1) -> bool:
    highs = df["high"]
    window = highs.iloc[-lookback:]
    recent_max_pos = window.values.argmax()
    bars_since = len(window) - 1 - recent_max_pos
    turning_down = float(df["close"].iloc[-1]) < float(df["close"].iloc[-2])
    return bars_since <= confirm and turning_down


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)).dropna()


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return log_returns(series).rolling(window).std()
