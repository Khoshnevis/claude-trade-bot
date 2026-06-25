"""Gaussian-HMM regime detector used to gate the trend engine.

Fits 2-3 hidden states on (return, realized-range) features and classifies the
current bar as 'trending' vs 'ranging' by the absolute mean return of its
state. Retrained periodically because transition probabilities are
non-stationary. Degrades to "always trending" if hmmlearn is unavailable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover
    GaussianHMM = None  # type: ignore

log = logging.getLogger("trade_bot.regime")

# hmmlearn emits noisy "Model is not converging" lines to the root logger on
# every refit. Non-convergence is harmless here (we cap iterations on purpose),
# so silence its logger to keep our output readable.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


class RegimeFilter:
    def __init__(self, n_states: int = 3, retrain_bars: int = 500,
                 lookback: int = 1000):
        self.n_states = n_states
        self.retrain_bars = retrain_bars
        self.lookback = lookback
        self._model = None
        self._trending_states: set[int] = set()
        self._bars_since_fit = 10 ** 9
        self._available = GaussianHMM is not None
        if not self._available:
            log.warning("hmmlearn not installed; trend regime gate disabled (always-on).")

    @staticmethod
    def _features(df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        ret = np.log(close / close.shift(1))
        rng = (df["high"] - df["low"]) / close
        feat = pd.concat([ret, rng], axis=1).dropna()
        return feat.to_numpy()

    def _fit(self, df: pd.DataFrame) -> None:
        sub = df.iloc[-self.lookback:]
        X = self._features(sub)
        if len(X) < max(50, self.n_states * 10):
            return
        try:
            model = GaussianHMM(
                n_components=self.n_states, covariance_type="diag",
                n_iter=100, random_state=42,
            )
            model.fit(X)
        except Exception as exc:  # noqa: BLE001
            log.warning("HMM fit failed: %s", exc)
            return
        # The state with the largest |mean return| is the trending state(s).
        mean_abs_ret = np.abs(model.means_[:, 0])
        thresh = mean_abs_ret.mean()
        self._trending_states = {i for i, m in enumerate(mean_abs_ret) if m >= thresh}
        self._model = model
        self._bars_since_fit = 0
        log.info("HMM refit: trending states=%s", sorted(self._trending_states))

    def is_trending(self, df: pd.DataFrame) -> bool:
        if not self._available:
            return True
        self._bars_since_fit += 1
        if self._model is None or self._bars_since_fit >= self.retrain_bars:
            self._fit(df)
        if self._model is None:
            return True
        X = self._features(df.iloc[-self.lookback:])
        if len(X) == 0:
            return True
        try:
            state = int(self._model.predict(X)[-1])
        except Exception as exc:  # noqa: BLE001
            log.warning("HMM predict failed: %s", exc)
            return True
        return state in self._trending_states
