"""Kalman-filter cointegration pairs engine (market-neutral mean-reversion).

For a pair (y, x) we model:  y_t = beta_t * x_t + alpha_t + e_t
The hidden state [alpha_t, beta_t] follows a random walk, so the hedge ratio
adapts through structural breaks without an arbitrary rolling window
(Chan / Palomar). The normalized forecast error of the Kalman filter is the
trading z-score; we enter on reversion at +/- entry_z and exit near zero.

A hand-rolled 2-state filter is used (no pykalman dependency). An optional
Engle-Granger cointegration gate (statsmodels) refuses to trade pairs whose
relationship has decayed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint
except Exception:  # pragma: no cover
    coint = None  # type: ignore

log = logging.getLogger("trade_bot.pairs")


@dataclass
class PairSignal:
    a: str               # y leg
    b: str               # x leg
    action: str          # 'enter_long' | 'enter_short' | 'exit' | 'hold'
    z: float
    beta: float          # hedge ratio: short beta units of x per unit of y
    spread_std: float    # std of forecast error, for risk sizing
    note: str = ""


class _KalmanHedge:
    """Recursive estimator of [alpha, beta] for y ~ beta*x + alpha."""

    def __init__(self, delta: float, ve: float):
        # Random-walk transition covariance for the state.
        self.Vw = (delta / (1.0 - delta)) * np.eye(2) if delta < 1 else delta * np.eye(2)
        self.ve = ve
        self.beta = np.zeros(2)          # [alpha, beta]
        self.P = np.zeros((2, 2))
        self.R = None
        self.initialized = False

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Feed one observation; return (forecast_error, forecast_std)."""
        obs = np.array([1.0, x])         # design row [1, x]
        if not self.initialized:
            self.beta = np.array([0.0, 1.0])
            self.R = np.zeros((2, 2))
            self.initialized = True
        # Predict
        self.R = self.P + self.Vw
        yhat = obs @ self.beta
        e = y - yhat                     # forecast error (the spread)
        Q = obs @ self.R @ obs + self.ve  # forecast error variance
        Q = max(Q, 1e-12)
        # Update
        K = (self.R @ obs) / Q           # Kalman gain
        self.beta = self.beta + K * e
        self.P = self.R - np.outer(K, obs) @ self.R
        return float(e), float(np.sqrt(Q))


class KalmanPairsEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.entry_z = cfg["entry_z"]
        self.exit_z = cfg["exit_z"]
        self.stop_z = cfg["stop_z"]
        self.warmup = cfg["warmup_bars"]

    def evaluate(self, a: str, b: str, df_a: pd.DataFrame,
                 df_b: pd.DataFrame) -> PairSignal | None:
        """Run the filter over aligned closes and emit a signal for the last bar."""
        joined = pd.concat(
            [df_a["close"].rename("y"), df_b["close"].rename("x")], axis=1
        ).dropna()
        if len(joined) < self.warmup + 10:
            log.debug("%s/%s: insufficient aligned history (%d bars)", a, b, len(joined))
            return None

        if self.cfg.get("recheck_coint") and not self._is_cointegrated(joined):
            return PairSignal(a, b, "hold", 0.0, 0.0, 0.0, note="not cointegrated")

        kf = _KalmanHedge(self.cfg["delta"], self.cfg["ve"])
        last_e = last_std = 0.0
        last_beta = 1.0
        ys = joined["y"].to_numpy()
        xs = joined["x"].to_numpy()
        for i in range(len(joined)):
            e, std = kf.update(xs[i], ys[i])
            last_e, last_std, last_beta = e, std, kf.beta[1]

        if last_std <= 0:
            return None
        z = last_e / last_std
        action = "hold"
        if abs(z) >= self.stop_z:
            action = "exit"          # blow-out stop
        elif z >= self.entry_z:
            action = "enter_short"   # spread rich -> short y, long x
        elif z <= -self.entry_z:
            action = "enter_long"    # spread cheap -> long y, short x
        elif abs(z) <= self.exit_z:
            action = "exit"
        return PairSignal(a, b, action, z, float(last_beta), float(last_std))

    def _is_cointegrated(self, joined: pd.DataFrame) -> bool:
        if coint is None:
            return True  # statsmodels missing -> don't block (logged once elsewhere)
        look = min(self.cfg.get("coint_lookback", 500), len(joined))
        sub = joined.iloc[-look:]
        try:
            _, pvalue, _ = coint(sub["y"], sub["x"])
        except Exception as exc:  # noqa: BLE001
            log.warning("coint test failed: %s", exc)
            return False
        return pvalue <= self.cfg.get("coint_pvalue", 0.05)
