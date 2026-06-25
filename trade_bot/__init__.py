"""Fully automated MT5 algorithmic forex trading bot.

Two uncorrelated low-frequency engines:
  1. Kalman / cointegration mean-reversion pairs (market-neutral core)
  2. Regime-filtered slow trend / momentum overlay

The MT5 terminal is assumed to be already logged in; the bot connects over the
local IPC bridge without credentials.
"""

__version__ = "0.1.0"
