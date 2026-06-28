"""Meta-labeling ML pipeline (López de Prado style).

The ML model does NOT predict price. A rule-based PRIMARY signal proposes a
side+timing (gold trends, so a breakout/momentum rule); the ML model is a
META-LABEL classifier that estimates P(this primary signal is profitable) from
features available at decision time, and we only take/size trades whose
probability clears a threshold.

Everything here is look-ahead safe: features at bar i use only data up to i,
and labels look forward only to assign the training target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- indicators (vectorized, causal) ----------------------------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l = df["high"], df["low"]
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(df, period).replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


# --- feature matrix ---------------------------------------------------------
FEATURE_COLUMNS = [
    "rsi", "adx", "atr_norm", "dist_ema_fast_atr", "dist_ema_slow_atr",
    "ema_spread_atr", "ret_5", "ret_20", "vol_20", "donch_pos", "hour_sin", "hour_cos",
]


def build_features(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                   donch: int = 20) -> pd.DataFrame:
    close = df["close"]
    atr = _atr(df).replace(0, np.nan)
    ema_f, ema_s = _ema(close, fast), _ema(close, slow)
    hh = df["high"].rolling(donch).max()
    ll = df["low"].rolling(donch).min()
    rng = (hh - ll).replace(0, np.nan)
    hour = df.index.hour
    feats = pd.DataFrame(index=df.index)
    feats["rsi"] = _rsi(close)
    feats["adx"] = _adx(df)
    feats["atr_norm"] = atr / close
    feats["dist_ema_fast_atr"] = (close - ema_f) / atr
    feats["dist_ema_slow_atr"] = (close - ema_s) / atr
    feats["ema_spread_atr"] = (ema_f - ema_s) / atr
    feats["ret_5"] = close.pct_change(5)
    feats["ret_20"] = close.pct_change(20)
    feats["vol_20"] = np.log(close / close.shift(1)).rolling(20).std()
    feats["donch_pos"] = ((close - ll) / rng).clip(0, 1)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return feats


# --- primary signal (rule-based side + timing) ------------------------------
def primary_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                   donch: int = 20) -> pd.Series:
    """+1 long / -1 short / 0 none, only on the rising edge of a breakout that
    agrees with the EMA trend (so signals are events, not every bar)."""
    close = df["close"]
    ema_f, ema_s = _ema(close, fast), _ema(close, slow)
    prior_high = df["high"].rolling(donch).max().shift(1)
    prior_low = df["low"].rolling(donch).min().shift(1)
    long_cond = (ema_f > ema_s) & (close > prior_high)
    short_cond = (ema_f < ema_s) & (close < prior_low)
    sig = pd.Series(0, index=df.index)
    sig[long_cond & ~long_cond.shift(1, fill_value=False)] = 1
    sig[short_cond & ~short_cond.shift(1, fill_value=False)] = -1
    return sig


# --- triple-barrier labels --------------------------------------------------
def triple_barrier_labels(df: pd.DataFrame, signals: pd.Series,
                          sl_atr: float = 1.5, tp_atr: float = 2.0,
                          horizon: int = 24) -> pd.DataFrame:
    """For each non-zero signal, label 1 if the take-profit barrier is hit
    before the stop-loss within `horizon` bars, else 0."""
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = _atr(df).to_numpy()
    idx = np.where(signals.to_numpy() != 0)[0]
    rows = []
    n = len(df)
    for i in idx:
        if i + 1 >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        side = int(signals.iloc[i])
        entry = close[i]
        tp = entry + side * tp_atr * atr[i]
        sl = entry - side * sl_atr * atr[i]
        label = None
        end = min(i + horizon, n - 1)
        for j in range(i + 1, end + 1):
            if side > 0:
                if low[j] <= sl:
                    label = 0; break
                if high[j] >= tp:
                    label = 1; break
            else:
                if high[j] >= sl:
                    label = 0; break
                if low[j] <= tp:
                    label = 1; break
        if label is None:  # neither barrier hit: label by final direction
            label = 1 if (close[end] - entry) * side > 0 else 0
        rows.append({"pos": i, "time": df.index[i], "side": side, "label": label})
    return pd.DataFrame(rows)


def build_dataset(df: pd.DataFrame, fast: int = 20, slow: int = 50,
                  donch: int = 20, sl_atr: float = 1.5, tp_atr: float = 2.0,
                  horizon: int = 24) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X features, y labels, meta) aligned on primary-signal bars."""
    feats = build_features(df, fast, slow, donch)
    sig = primary_signal(df, fast, slow, donch)
    labels = triple_barrier_labels(df, sig, sl_atr, tp_atr, horizon)
    if labels.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=int), labels
    rows = labels["pos"].to_numpy()
    X = feats.iloc[rows].copy()
    X["side"] = labels["side"].to_numpy()  # the side is itself a feature
    y = pd.Series(labels["label"].to_numpy(), index=X.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    return X[mask], y[mask.to_numpy()], labels[mask.to_numpy()]
