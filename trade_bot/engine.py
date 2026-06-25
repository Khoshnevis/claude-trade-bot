"""Main orchestration loop.

Event cadence: act once per closed bar of the configured timeframe. Each cycle:
  1. ensure the MT5 link is alive
  2. refresh equity, high-water mark, daily anchor
  3. evaluate the drawdown kill-switch (flatten + halt if breached)
  4. unless halted / in a blackout window, run both engines
  5. periodically reconcile owned positions against the broker

The loop is idempotent and wrapped so a single bad cycle never kills the bot.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from .config import Config
from .execution import Executor, _ok, pair_tag, reversal_tag, trend_tag
from .features import atr
from .mt5_client import MT5Client, timeframe_const
from .persistence import StateStore
from .risk import RiskManager
from .signals import KalmanPairsEngine, ReversalEngine, TrendEngine

log = logging.getLogger("trade_bot.engine")


class TradingEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = MT5Client(
            terminal_path=cfg.mt5["terminal_path"],
            magic=cfg.mt5["magic"],
            deviation_points=cfg.mt5["deviation_points"],
        )
        self.executor = Executor(self.client)
        self.risk = RiskManager(cfg.risk, cfg.get("correlation_clusters", []))
        self.state = StateStore(cfg["persistence"]["db_path"])
        self.symbols = cfg.all_symbols()
        self.tf_name = cfg.engine["timeframe"]
        self.tf = None  # resolved after connect
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
        # Soft SL/TP levels for reversal positions, keyed by position ticket.
        # Persisted so a restart keeps monitoring them.
        self.soft_stops: dict[int, dict] = {}
        # Higher "context" timeframe for trend MTF confirmation (optional).
        self.context_tf_name = (
            cfg.trend.get("context_timeframe") if cfg.trend.get("enabled") else None
        )
        self.context_tf = None  # resolved after connect
        self._last_bar_time = None
        self._last_reconcile = 0.0
        self._data: dict[str, pd.DataFrame] = {}
        self._context_data: dict[str, pd.DataFrame] = {}

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        self.client.connect()
        self.tf = timeframe_const(self.tf_name)
        if self.context_tf_name:
            self.context_tf = timeframe_const(self.context_tf_name)
        self.client.ensure_symbols(self.symbols)
        self._load_soft_stops()
        log.info("Engine started | tf=%s | context_tf=%s | symbols=%s",
                 self.tf_name, self.context_tf_name, self.symbols)
        self._loop()

    def stop(self) -> None:
        self.state.close()
        self.client.shutdown()

    # -- main loop ------------------------------------------------------------
    def _loop(self) -> None:
        poll = self.cfg.engine["poll_seconds"]
        while True:
            try:
                self.client.ensure_connected()
                # Soft SL/TP are checked on EVERY poll (intrabar), not just at
                # bar close, so the bot reacts to price within the hour.
                if self.reversal_engine:
                    self._monitor_soft_stops()
                if self._new_bar_ready():
                    self._on_bar()
            except KeyboardInterrupt:
                log.info("Interrupted; shutting down.")
                break
            except Exception as exc:  # noqa: BLE001
                log.exception("Cycle error: %s", exc)
            time.sleep(poll)

    def _new_bar_ready(self) -> bool:
        ref = self.symbols[0]
        df = self.client.rates(ref, self.tf, 3)
        latest_closed = df.index[-2]  # last fully closed bar
        if self._last_bar_time is None or latest_closed > self._last_bar_time:
            self._last_bar_time = latest_closed
            return True
        return False

    def _on_bar(self) -> None:
        log.info("New %s bar closed at %s", self.tf_name, self._last_bar_time)
        equity = self.client.equity()
        hw = self.state.high_water(equity)
        day_anchor, is_new_day = self.state.day_anchor(equity)
        if is_new_day:
            self.risk.reset_daily()

        # Kill-switch evaluation.
        flatten, reason = self.risk.check_drawdown(equity, hw, day_anchor)
        if flatten:
            log.critical("Kill-switch flatten triggered: %s", reason)
            self._flatten_all()
            self.state.log_event("kill_switch", {"reason": reason, "equity": equity})
            return
        if self.risk.state.halted:
            log.warning("Halted (%s); managing exits only.", self.risk.state.halt_reason)

        # Pull data once per cycle for all symbols.
        self._refresh_data()

        if not self._in_blackout():
            if self.pairs_engine:
                self._run_pairs(equity)
            if self.trend_engine:
                self._run_trend(equity)
            if self.reversal_engine:
                self._run_reversal(equity)
        else:
            log.info("In blackout window; skipping new entries.")

        # Always manage exits regardless of blackout/halt.
        if self.trend_engine:
            self._manage_trend_exits()
        if self.reversal_engine:
            self._manage_reversal_exits()

        self._maybe_reconcile()

    # -- data -----------------------------------------------------------------
    def _refresh_data(self) -> None:
        n = self.cfg.engine["history_bars"]
        for sym in self.symbols:
            try:
                self._data[sym] = self.client.rates(sym, self.tf, n)
            except Exception as exc:  # noqa: BLE001
                log.error("Data fetch failed for %s: %s", sym, exc)
        # Context-timeframe bars for the trend engine's MTF direction/regime.
        if self.context_tf is not None:
            for sym in self.cfg.trend.get("symbols", []):
                try:
                    self._context_data[sym] = self.client.rates(sym, self.context_tf, n)
                except Exception as exc:  # noqa: BLE001
                    log.error("Context data fetch failed for %s: %s", sym, exc)

    # -- pairs engine ---------------------------------------------------------
    def _run_pairs(self, equity: float) -> None:
        cfg = self.cfg.pairs
        for p in cfg["pairs"]:
            a, b = p["a"], p["b"]
            df_a, df_b = self._data.get(a), self._data.get(b)
            if df_a is None or df_b is None:
                continue
            sig = self.pairs_engine.evaluate(a, b, df_a, df_b)
            if sig is None:
                continue
            held = self.executor.has_pair(a, b)
            log.info("PAIR %s/%s z=%.2f action=%s held=%s %s",
                     a, b, sig.z, sig.action, held, sig.note)

            if held and sig.action in ("exit", "hold"):
                if sig.action == "exit":
                    self.executor.close_pair(a, b)
                    self.state.log_event("pair_exit", {"a": a, "b": b, "z": sig.z})
                continue

            if not held and sig.action in ("enter_long", "enter_short"):
                if self.risk.state.halted:
                    continue
                self._open_pair_sized(a, b, sig, equity)

    def _open_pair_sized(self, a: str, b: str, sig, equity: float) -> None:
        ok, why = self.risk.can_open(a, sig.action, self.client.positions(),
                                     self.cfg.risk["risk_per_trade"])
        if not ok:
            log.info("Pair %s/%s blocked: %s", a, b, why)
            return
        info_a = self.client.symbol_info(a)
        info_b = self.client.symbol_info(b)
        # Stop distance (in y price) implied by the z-score blow-out stop.
        stop_dist = max(
            (self.cfg.pairs["stop_z"] - self.cfg.pairs["entry_z"]) * sig.spread_std,
            sig.spread_std,
        )
        lots_y = self.risk.lots_for_risk(info_a, equity, stop_dist)
        if lots_y <= 0:
            log.info("Pair %s/%s: y-leg too small to size within risk.", a, b)
            return
        beta = abs(sig.beta) if sig.beta else 1.0
        lots_x = self.risk._round_volume(info_b, lots_y * beta)
        if lots_x <= 0:
            lots_x = getattr(info_b, "volume_min", 0.01)
        direction = "long" if sig.action == "enter_long" else "short"
        if self.executor.open_pair(a, b, direction, lots_y, lots_x):
            self.state.log_event("pair_enter", {
                "a": a, "b": b, "dir": direction, "z": sig.z,
                "beta": sig.beta, "lots_y": lots_y, "lots_x": lots_x,
            })

    # -- trend engine ---------------------------------------------------------
    def _run_trend(self, equity: float) -> None:
        for sym in self.cfg.trend["symbols"]:
            df = self._data.get(sym)
            if df is None:
                continue
            sig = self.trend_engine.evaluate(sym, df, self._context_data.get(sym))
            if sig is None:
                continue
            if sig.action not in ("enter_long", "enter_short"):
                continue
            if self.executor.has_trend(sym):
                continue
            if self.risk.state.halted:
                continue
            side = "buy" if sig.action == "enter_long" else "sell"
            ok, why = self.risk.can_open(sym, side, self.client.positions(),
                                         self.cfg.risk["risk_per_trade"])
            if not ok:
                log.info("Trend %s blocked: %s", sym, why)
                continue
            self._open_trend_sized(sym, side, sig, equity)

    def _open_trend_sized(self, sym: str, side: str, sig, equity: float) -> None:
        info = self.client.symbol_info(sym)
        tick = self.client.tick(sym)
        if info is None or tick is None or sig.atr <= 0:
            return
        stop_dist = self.cfg.risk["atr_stop_mult"] * sig.atr
        tp_dist = self.cfg.risk["atr_target_mult"] * sig.atr
        stop_dist = self._respect_min_stop(info, stop_dist)
        price = tick.ask if side == "buy" else tick.bid
        if side == "buy":
            sl, tp = price - stop_dist, price + tp_dist
        else:
            sl, tp = price + stop_dist, price - tp_dist
        lots = self.risk.lots_for_risk(info, equity, stop_dist)
        if lots <= 0:
            log.info("Trend %s: too small to size within risk.", sym)
            return
        if self.executor.open_trend(sym, side, lots, round(sl, info.digits),
                                    round(tp, info.digits)):
            self.state.log_event("trend_enter", {
                "symbol": sym, "side": side, "lots": lots,
                "sl": sl, "tp": tp, "atr": sig.atr,
            })

    def _manage_trend_exits(self) -> None:
        for sym in self.cfg.trend["symbols"]:
            pos = self.executor.trend_position(sym)
            if pos is None:
                continue
            df = self._data.get(sym)
            if df is None:
                continue
            sig = self.trend_engine.evaluate(sym, df, self._context_data.get(sym))
            if sig is None:
                continue
            is_long = pos.type == 0
            if (is_long and sig.action == "exit_long_if_held") or (
                not is_long and sig.action == "exit_short_if_held"
            ):
                log.info("Trend %s: MA flip exit.", sym)
                self.executor.close_tagged(trend_tag(sym))
                self.state.log_event("trend_exit", {"symbol": sym})

    # -- reversal engine (soft SL/TP) ----------------------------------------
    def _run_reversal(self, equity: float) -> None:
        for sym in self.cfg.reversal["symbols"]:
            df = self._data.get(sym)
            if df is None:
                continue
            # No pyramiding: skip if we already hold the max for this symbol.
            held = self.executor.owned_tags().get(reversal_tag(sym), [])
            if len(held) >= self.cfg.reversal.get("max_per_symbol", 1):
                continue
            if self.risk.state.halted:
                continue
            sig = self.reversal_engine.evaluate(sym, df)
            if sig is None or sig.action not in ("enter_long", "enter_short"):
                continue
            side = "buy" if sig.action == "enter_long" else "sell"
            ok, why = self.risk.can_open(sym, side, self.client.positions(),
                                         self.cfg.risk["risk_per_trade"])
            if not ok:
                log.info("Reversal %s blocked: %s", sym, why)
                continue
            self._open_reversal_sized(sym, side, sig, equity)

    def _open_reversal_sized(self, sym: str, side: str, sig, equity: float) -> None:
        info = self.client.symbol_info(sym)
        tick = self.client.tick(sym)
        if info is None or tick is None or sig.atr <= 0:
            return
        stop_dist = self.cfg.reversal["stop_atr"] * sig.atr
        tp_dist = self.cfg.reversal["target_atr"] * sig.atr
        lots = self.risk.lots_for_risk(info, equity, stop_dist)
        if lots <= 0:
            log.info("Reversal %s: too small to size within risk.", sym)
            return
        price = tick.ask if side == "buy" else tick.bid
        s = 1 if side == "buy" else -1
        soft_sl = price - s * stop_dist
        soft_tp = price + s * tp_dist
        res = self.executor.open_reversal(sym, side, lots)
        if res is None or getattr(res, "retcode", None) != 10009:
            return
        ticket = self._resolve_ticket(sym, res)
        if ticket is not None:
            self.soft_stops[ticket] = {
                "symbol": sym, "side": s, "sl": soft_sl, "tp": soft_tp,
            }
            self._save_soft_stops()
        self.state.log_event("reversal_enter", {
            "symbol": sym, "side": side, "lots": lots, "rsi": sig.rsi,
            "soft_sl": soft_sl, "soft_tp": soft_tp, "ticket": ticket,
        })

    def _resolve_ticket(self, sym: str, res) -> int | None:
        """Map the order result to the open position ticket."""
        order_ticket = getattr(res, "order", None)
        for p in self.client.positions(magic_only=True):
            if p.symbol == sym and (p.ticket == order_ticket
                                    or reversal_tag(sym) == (p.comment or "")):
                if p.ticket not in self.soft_stops:
                    return p.ticket
        return order_ticket

    def _manage_reversal_exits(self) -> None:
        """Mean-reversion (soft TP) exits, evaluated at bar close."""
        owned = self.executor.owned_tags()
        for sym in self.cfg.reversal["symbols"]:
            positions = owned.get(reversal_tag(sym), [])
            if not positions:
                continue
            df = self._data.get(sym)
            if df is None:
                continue
            for p in positions:
                side = 1 if p.type == 0 else -1
                if self.reversal_engine.reversion_done(df, side):
                    log.info("Reversal %s: mean-reversion exit (ticket %s).",
                             sym, p.ticket)
                    if _ok(self.client.close_position(p)):
                        self._drop_soft_stop(p.ticket)
                        self.state.log_event("reversal_exit",
                                             {"symbol": sym, "ticket": p.ticket})

    def _monitor_soft_stops(self) -> None:
        """Intrabar SL/TP check using live ticks (runs every poll)."""
        if not self.soft_stops:
            return
        open_tickets = {p.ticket for p in self.client.positions(magic_only=True)}
        for ticket in list(self.soft_stops):
            if ticket not in open_tickets:
                self._drop_soft_stop(ticket)  # closed elsewhere; forget it
                continue
            lvl = self.soft_stops[ticket]
            tick = self.client.tick(lvl["symbol"])
            if tick is None:
                continue
            # Long exits on bid; short exits on ask.
            price = tick.bid if lvl["side"] > 0 else tick.ask
            hit_sl = (price <= lvl["sl"]) if lvl["side"] > 0 else (price >= lvl["sl"])
            hit_tp = (price >= lvl["tp"]) if lvl["side"] > 0 else (price <= lvl["tp"])
            if hit_sl or hit_tp:
                pos = next((p for p in self.client.positions(magic_only=True)
                            if p.ticket == ticket), None)
                if pos is not None and _ok(self.client.close_position(pos)):
                    reason = "soft_sl" if hit_sl else "soft_tp"
                    log.info("Reversal %s: %s hit @%.5f (ticket %s).",
                             lvl["symbol"], reason, price, ticket)
                    self.state.log_event(reason, {"symbol": lvl["symbol"],
                                                  "ticket": ticket, "price": price})
                    self._drop_soft_stop(ticket)

    def _load_soft_stops(self) -> None:
        stored = self.state.get("soft_stops", {})
        self.soft_stops = {int(k): v for k, v in stored.items()}
        if self.soft_stops:
            log.info("Loaded %d soft-stop levels from state.", len(self.soft_stops))

    def _save_soft_stops(self) -> None:
        self.state.set("soft_stops", {str(k): v for k, v in self.soft_stops.items()})

    def _drop_soft_stop(self, ticket: int) -> None:
        if ticket in self.soft_stops:
            del self.soft_stops[ticket]
            self._save_soft_stops()

    @staticmethod
    def _respect_min_stop(info, stop_dist: float) -> float:
        point = getattr(info, "point", 0.0)
        min_pts = getattr(info, "trade_stops_level", 0)
        if point > 0 and min_pts > 0:
            return max(stop_dist, min_pts * point * 1.2)
        return stop_dist

    # -- housekeeping ---------------------------------------------------------
    def _flatten_all(self) -> None:
        for p in self.client.positions(magic_only=True):
            self.client.close_position(p)

    def _in_blackout(self) -> bool:
        now = datetime.now(timezone.utc).strftime("%H:%M")
        for window in self.cfg.engine.get("blackouts", []):
            start, end = window.split("-")
            if start <= end:
                if start <= now <= end:
                    return True
            else:  # wraps midnight
                if now >= start or now <= end:
                    return True
        return False

    def _maybe_reconcile(self) -> None:
        now = time.time()
        if now - self._last_reconcile < self.cfg.engine["reconcile_seconds"]:
            return
        self._last_reconcile = now
        owned = self.executor.owned_tags()
        n = sum(len(v) for v in owned.values())
        log.info("Reconcile: %d owned positions | tags=%s",
                 n, {k: len(v) for k, v in owned.items() if k})
        # Detect a half-open pair (one leg only) and flatten it to stay neutral.
        for p in self.cfg.pairs.get("pairs", []):
            a, b = p["a"], p["b"]
            y = owned.get(pair_tag(a, b, "y"))
            x = owned.get(pair_tag(a, b, "x"))
            if bool(y) != bool(x):
                log.warning("Pair %s/%s half-open; flattening leg.", a, b)
                self.executor.close_pair(a, b)
