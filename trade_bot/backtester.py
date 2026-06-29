"""Event-driven backtester that replays the SAME signal engines used live.

Design goals:
  * Single source of truth: it calls KalmanPairsEngine / TrendEngine exactly as
    the live engine does, on expanding windows, so research == live.
  * No look-ahead: at each closed bar it decides using data up to that bar only;
    entries fill at that bar's close, stops/targets are checked on later bars.
  * Honest costs: spread + commission + slippage are charged on every round
    turn via the configurable `backtest` cost model.

This is intentionally data-source-agnostic (it takes plain DataFrames + symbol
metadata) so it can be driven from MT5 history or tested on synthetic data.

A single backtest path is weak evidence (Quantopian: in-sample Sharpe predicts
out-of-sample with R^2 < 0.025). Use it to KILL bad configs, not bless good ones.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .risk import RiskManager
from .signals import KalmanPairsEngine, ReversalEngine, TrendEngine

log = logging.getLogger("trade_bot.backtest")

# Approximate base-timeframe bars per year, for annualizing the Sharpe ratio.
_BARS_PER_YEAR = {
    "M5": 12 * 24 * 5 * 52, "M15": 4 * 24 * 5 * 52, "M30": 2 * 24 * 5 * 52,
    "H1": 24 * 5 * 52, "H4": 6 * 5 * 52, "D1": 252, "W1": 52,
}


def pip_size(digits: int, point: float) -> float:
    """Pip size: 10*point on fractional-pip (3/5-digit) symbols, else point."""
    return point * 10 if digits in (3, 5) else point


@dataclass
class Position:
    symbol: str
    side: int            # +1 long, -1 short
    volume: float
    entry: float
    sl: float | None
    tp: float | None
    kind: str            # 'trend' | 'pair' | 'reversal'
    open_i: int
    pair_id: str = ""
    peak: float = 0.0    # most-favorable price seen (for trailing stops)
    atr0: float = 0.0    # ATR at entry (fixes the trailing distance)

    @property
    def type(self) -> int:
        """MT5-compatible side: ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1.

        Lets RiskManager.can_open treat sim positions like live ones.
        """
        return 0 if self.side > 0 else 1


@dataclass
class Trade:
    symbol: str
    kind: str
    side: int
    volume: float
    entry: float
    exit: float
    pnl: float
    open_i: int
    close_i: int
    reason: str


@dataclass
class Result:
    start_equity: float
    end_equity: float
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    diag: dict = field(default_factory=dict)
    warmup: int = 0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def metrics(self, bars_per_year: float) -> dict:
        ec = self.equity_curve
        ret = ec.pct_change().dropna()
        total = self.end_equity / self.start_equity - 1
        if len(ret) > 1 and ret.std() > 0:
            sharpe = (ret.mean() / ret.std()) * math.sqrt(bars_per_year)
        else:
            sharpe = 0.0
        peak = ec.cummax()
        max_dd = float(((ec - peak) / peak).min()) if len(ec) else 0.0
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = -sum(t.pnl for t in losses)
        win_rate = len(wins) / self.n_trades if self.n_trades else 0.0
        avg_win = gross_win / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0
        expectancy = (sum(t.pnl for t in self.trades) / self.n_trades
                      if self.n_trades else 0.0)
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        return {
            "total_return": total,
            "end_equity": self.end_equity,
            "sharpe_annual": sharpe,
            "max_drawdown": max_dd,
            "n_trades": self.n_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
        }


class Backtester:
    def __init__(self, cfg, data: dict[str, pd.DataFrame],
                 context_data: dict[str, pd.DataFrame],
                 meta: dict[str, SimpleNamespace], bars_per_year: float):
        self.cfg = cfg
        self.data = data
        self.context_data = context_data
        self.meta = meta
        self.bars_per_year = bars_per_year
        self.bt = cfg["backtest"]
        self.window = self.bt["history_window"]
        self.risk = RiskManager(cfg.risk, cfg.get("correlation_clusters", []))
        self.pairs_engine = (
            KalmanPairsEngine(cfg.pairs) if cfg.pairs.get("enabled") else None
        )
        self.trend_engine = (
            TrendEngine(cfg.trend, atr_period=cfg.risk["atr_period"])
            if cfg.trend.get("enabled")
            else None
        )
        self.reversal_engine = (
            ReversalEngine(cfg.reversal, atr_period=cfg.risk["atr_period"])
            if cfg.reversal.get("enabled")
            else None
        )
        # Reference timeline = the most common base index.
        self.ref = next(iter(data))
        self.index = data[self.ref].index
        # Why-no-trade diagnostics (tallied across the replay).
        self.diag: dict[str, int] = defaultdict(int)
        # Research mode: skip the permanent drawdown halt so the FULL period
        # (and a meaningful OOS half) is evaluated instead of truncated.
        self.disable_killswitch = False

    # -- cost model -----------------------------------------------------------
    def _round_turn_cost(self, symbol: str, volume: float) -> float:
        m = self.meta[symbol]
        ps = pip_size(m.digits, m.point)
        spreads = self.bt["spread_pips"]
        spread_pips = spreads.get(symbol, spreads.get("default", 0.8))
        slip = self.bt["slippage_pips"]
        # spread once + slippage on both sides, in price -> money via tick value.
        price_cost = (spread_pips + 2 * slip) * ps
        money = (price_cost / m.trade_tick_size) * m.trade_tick_value * volume
        money += 2 * self.bt["commission_per_lot_per_side"] * volume
        return money

    def _pnl(self, symbol: str, side: int, volume: float,
             entry: float, exit: float) -> float:
        m = self.meta[symbol]
        gross = side * (exit - entry)
        money = (gross / m.trade_tick_size) * m.trade_tick_value * volume
        return money - self._round_turn_cost(symbol, volume)

    # -- helpers --------------------------------------------------------------
    def _slice(self, frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
        sub = frame[frame.index <= t]
        return sub.iloc[-self.window:]

    def _ctx_slice(self, symbol: str, t: pd.Timestamp) -> pd.DataFrame | None:
        cdf = self.context_data.get(symbol)
        if cdf is None:
            return None
        sub = cdf[cdf.index <= t]
        return sub.iloc[-self.window:] if len(sub) else None

    # -- main replay ----------------------------------------------------------
    def run(self) -> Result:
        start_eq = self.bt["start_equity"]
        realized = 0.0
        positions: list[Position] = []
        trades: list[Trade] = []
        curve: list[float] = []
        curve_idx: list[pd.Timestamp] = []

        warmup = max(self.window, self.cfg.pairs.get("warmup_bars", 250)) + 5
        n = len(self.index)
        cur_day = None
        day_start_eq = start_eq
        log.info("Replaying %d bars (warmup=%d)...", n - warmup, warmup)

        for i in range(warmup, n):
            t = self.index[i]
            if (i - warmup) % 500 == 0 and i > warmup:
                log.info("  progress %d/%d bars | equity=%.2f | trades=%d",
                         i - warmup, n - warmup, curve[-1] if curve else start_eq,
                         len(trades))

            # 1) Manage open trend/reversal stops/targets against THIS bar's range.
            positions, stop_pnl = self._process_stops(positions, trades, t, i)
            realized += stop_pnl

            # 2) Mark-to-market equity (realized + open unrealized at close).
            equity = start_eq + realized + self._unrealized(positions, t)
            curve.append(equity)
            curve_idx.append(t)

            # 3) Per-day anchor: reset the daily-loss halt at each new UTC day.
            day = t.date()
            if day != cur_day:
                cur_day = day
                day_start_eq = equity
                self.risk.reset_daily()

            # 4) Kill-switch check on the simulated equity curve (unless research
            # mode disables it so we can see the full-period / OOS result).
            if not self.disable_killswitch:
                hw = max(curve)
                self.risk.check_drawdown(equity, hw, day_start_eq)
                if self.risk.state.halted:
                    realized += self._flatten(positions, trades, t, i, "kill_switch")
                    positions = []
                    continue

            # 5) Engines (decisions at this bar's close).
            if self.pairs_engine:
                realized += self._step_pairs(positions, trades, t, i, equity)
            if self.trend_engine:
                realized += self._step_trend(positions, trades, t, i, equity)
            if self.reversal_engine:
                realized += self._step_reversal(positions, trades, t, i, equity)

        # Close anything still open at the final bar.
        realized += self._flatten(positions, trades, self.index[-1], n - 1, "end")
        end_eq = start_eq + realized
        return Result(start_eq, end_eq, pd.Series(curve, index=curve_idx),
                      trades, dict(self.diag), warmup)

    # -- position management --------------------------------------------------
    def _unrealized(self, positions: list[Position], t: pd.Timestamp) -> float:
        total = 0.0
        for p in positions:
            px = float(self.data[p.symbol].loc[:t]["close"].iloc[-1])
            m = self.meta[p.symbol]
            total += (p.side * (px - p.entry) / m.trade_tick_size) * m.trade_tick_value * p.volume
        return total

    def _process_stops(self, positions, trades, t, i) -> tuple[list[Position], float]:
        survivors = []
        realized = 0.0
        for p in positions:
            if p.kind not in ("trend", "reversal") or (p.sl is None and p.tp is None):
                survivors.append(p)
                continue
            # Use the bar at-or-before t: symbols can miss a timestamp the
            # reference symbol has (holidays / differing session gaps).
            sub = self.data[p.symbol].loc[:t]
            if sub.empty:
                survivors.append(p)
                continue
            bar = sub.iloc[-1]
            hi, lo = float(bar["high"]), float(bar["low"])
            # Trailing (chandelier) stop for trend trades: ratchet the stop
            # behind the most-favorable price so winners can run.
            if p.kind == "trend" and self.cfg.trend.get("use_trailing") and p.atr0 > 0:
                trail = self.cfg.trend.get("trail_atr_mult", 3.0) * p.atr0
                if p.side > 0:
                    p.peak = max(p.peak, hi)
                    p.sl = max(p.sl, p.peak - trail)
                else:
                    p.peak = min(p.peak, lo)
                    p.sl = min(p.sl, p.peak + trail)
            exit_px = None
            reason = ""
            if p.side > 0:
                if p.sl is not None and lo <= p.sl:
                    exit_px, reason = p.sl, "sl"
                elif p.tp is not None and hi >= p.tp:
                    exit_px, reason = p.tp, "tp"
            else:
                if p.sl is not None and hi >= p.sl:
                    exit_px, reason = p.sl, "sl"
                elif p.tp is not None and lo <= p.tp:
                    exit_px, reason = p.tp, "tp"
            if exit_px is None:
                survivors.append(p)
            else:
                realized += self._book(p, exit_px, t, i, reason, trades)
        return survivors, realized

    def _book(self, p: Position, exit_px: float, t, i, reason, trades) -> float:
        pnl = self._pnl(p.symbol, p.side, p.volume, p.entry, exit_px)
        trades.append(Trade(p.symbol, p.kind, p.side, p.volume, p.entry,
                            exit_px, pnl, p.open_i, i, reason))
        return pnl

    def _flatten(self, positions, trades, t, i, reason) -> float:
        pnl = 0.0
        for p in positions:
            px = float(self.data[p.symbol].loc[:t]["close"].iloc[-1])
            pnl += self._book(p, px, t, i, reason, trades)
        return pnl

    # -- pairs ----------------------------------------------------------------
    def _step_pairs(self, positions, trades, t, i, equity) -> float:
        pnl = 0.0
        for pcfg in self.cfg.pairs["pairs"]:
            a, b = pcfg["a"], pcfg["b"]
            if a not in self.data or b not in self.data:
                continue
            pid = f"{a}{b}"
            legs = [p for p in positions if p.pair_id == pid]
            df_a, df_b = self._slice(self.data[a], t), self._slice(self.data[b], t)
            sig = self.pairs_engine.evaluate(a, b, df_a, df_b)
            if sig is None:
                continue
            # Diagnostics: why is/isn't this pair trading?
            if sig.note == "not cointegrated":
                self.diag["pair_coint_block"] += 1
            elif sig.action in ("enter_long", "enter_short"):
                self.diag["pair_entry_signal"] += 1
            else:
                self.diag["pair_cointegrated_no_entry"] += 1
            if legs and sig.action in ("exit", "enter_long", "enter_short"):
                # close on exit/stop OR on a flip; re-entry handled next bars
                if sig.action == "exit":
                    for p in legs:
                        pnl += self._book(p, self._px(p.symbol, t), t, i, "z_exit", trades)
                        positions.remove(p)
                continue
            if not legs and sig.action in ("enter_long", "enter_short") and not self.risk.state.halted:
                ok, _ = self.risk.can_open(a, sig.action, positions, self.cfg.risk["risk_per_trade"])
                if not ok:
                    self.diag["pair_blocked_by_risk"] += 1
                    continue
                if not self._open_pair(positions, a, b, pid, sig, t, i, equity):
                    self.diag["pair_size_skip"] += 1
                else:
                    self.diag["pair_opened"] += 1
        return pnl

    def _open_pair(self, positions, a, b, pid, sig, t, i, equity) -> bool:
        stop_dist = max((self.cfg.pairs["stop_z"] - self.cfg.pairs["entry_z"]) * sig.spread_std,
                        sig.spread_std)
        lots_y = self.risk.lots_for_risk(self.meta[a], equity, stop_dist)
        if lots_y <= 0:
            return False
        beta = abs(sig.beta) if sig.beta else 1.0
        lots_x = self.risk._round_volume(self.meta[b], lots_y * beta)
        if lots_x <= 0:
            lots_x = self.meta[b].volume_min
        side_y = 1 if sig.action == "enter_long" else -1
        positions.append(Position(a, side_y, lots_y, self._px(a, t), None, None, "pair", i, pid))
        positions.append(Position(b, -side_y, lots_x, self._px(b, t), None, None, "pair", i, pid))
        return True

    # -- trend ----------------------------------------------------------------
    def _step_trend(self, positions, trades, t, i, equity) -> float:
        pnl = 0.0
        for sym in self.cfg.trend["symbols"]:
            if sym not in self.data:
                continue
            held = [p for p in positions if p.kind == "trend" and p.symbol == sym]
            sig = self.trend_engine.evaluate(sym, self._slice(self.data[sym], t),
                                             self._ctx_slice(sym, t))
            if sig is None:
                continue
            # Diagnostics for the trend gate.
            if sig.note == "regime=ranging":
                self.diag["trend_regime_block"] += 1
            elif sig.action in ("enter_long", "enter_short"):
                self.diag["trend_entry_signal"] += 1
            else:
                self.diag["trend_no_setup"] += 1
            if held:
                p = held[0]
                if (p.side > 0 and sig.action == "exit_long_if_held") or \
                   (p.side < 0 and sig.action == "exit_short_if_held"):
                    pnl += self._book(p, self._px(sym, t), t, i, "ma_flip", trades)
                    positions.remove(p)
                continue
            if sig.action in ("enter_long", "enter_short") and not self.risk.state.halted:
                side = "buy" if sig.action == "enter_long" else "sell"
                ok, _ = self.risk.can_open(sym, side, positions, self.cfg.risk["risk_per_trade"])
                if not ok or sig.atr <= 0:
                    self.diag["trend_blocked_by_risk"] += 1
                    continue
                if not self._open_trend(positions, sym, sig, t, i, equity):
                    self.diag["trend_size_skip"] += 1
                else:
                    self.diag["trend_opened"] += 1
        return pnl

    def _open_trend(self, positions, sym, sig, t, i, equity) -> bool:
        m = self.meta[sym]
        stop_dist = self.cfg.risk["atr_stop_mult"] * sig.atr
        tp_dist = self.cfg.risk["atr_target_mult"] * sig.atr
        lots = self.risk.lots_for_risk(m, equity, stop_dist)
        if lots <= 0:
            return False
        price = self._px(sym, t)
        side = 1 if sig.action == "enter_long" else -1
        sl = price - side * stop_dist
        trailing = self.cfg.trend.get("use_trailing")
        tp = None if trailing else price + side * tp_dist
        positions.append(Position(sym, side, lots, price, sl, tp, "trend", i,
                                  peak=price, atr0=sig.atr))
        return True

    # -- reversal -------------------------------------------------------------
    def _step_reversal(self, positions, trades, t, i, equity) -> float:
        pnl = 0.0
        max_per = self.cfg.reversal.get("max_per_symbol", 1)
        for sym in self.cfg.reversal["symbols"]:
            if sym not in self.data:
                continue
            held = [p for p in positions if p.kind == "reversal" and p.symbol == sym]
            df = self._slice(self.data[sym], t)
            # Mean-reversion (soft TP) exit on held positions.
            for p in list(held):
                if self.reversal_engine.reversion_done(df, p.side):
                    pnl += self._book(p, self._px(sym, t), t, i, "reversion", trades)
                    positions.remove(p)
                    held.remove(p)
            if len(held) >= max_per:
                continue
            sig = self.reversal_engine.evaluate(sym, df)
            if sig is None:
                continue
            if sig.action in ("enter_long", "enter_short"):
                self.diag["reversal_entry_signal"] += 1
            else:
                self.diag["reversal_no_setup"] += 1
                continue
            if self.risk.state.halted:
                continue
            side = "buy" if sig.action == "enter_long" else "sell"
            ok, _ = self.risk.can_open(sym, side, positions, self.cfg.risk["risk_per_trade"])
            if not ok:
                self.diag["reversal_blocked_by_risk"] += 1
                continue
            if not self._open_reversal(positions, sym, sig, t, i, equity):
                self.diag["reversal_size_skip"] += 1
            else:
                self.diag["reversal_opened"] += 1
        return pnl

    def _open_reversal(self, positions, sym, sig, t, i, equity) -> bool:
        m = self.meta[sym]
        stop_dist = self.cfg.reversal["stop_atr"] * sig.atr
        tp_dist = self.cfg.reversal["target_atr"] * sig.atr
        lots = self.risk.lots_for_risk(m, equity, stop_dist)
        if lots <= 0:
            return False
        price = self._px(sym, t)
        side = 1 if sig.action == "enter_long" else -1
        sl = price - side * stop_dist
        tp = price + side * tp_dist
        positions.append(Position(sym, side, lots, price, sl, tp, "reversal", i))
        return True

    def _px(self, symbol: str, t: pd.Timestamp) -> float:
        return float(self.data[symbol].loc[:t]["close"].iloc[-1])
