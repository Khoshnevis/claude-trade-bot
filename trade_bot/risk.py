"""Risk layer: position sizing, portfolio limits, and the drawdown kill-switch.

Has veto power over every order. Sizing is risk-driven (fixed-fractional with
an ATR/spread-derived stop), never margin-driven.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger("trade_bot.risk")


@dataclass
class RiskState:
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, cfg: dict, clusters: list[list[str]]):
        self.cfg = cfg
        self.clusters = clusters
        self.state = RiskState()

    # -- sizing ---------------------------------------------------------------
    def lots_for_risk(self, symbol_info, equity: float, stop_price_dist: float,
                      risk_fraction: float | None = None) -> float:
        """Volume (lots) so that hitting the stop loses `risk_fraction` of equity.

        Uses the symbol's tick value/size so it is correct for any quote
        currency and contract size. Returns 0.0 if it cannot be sized safely.
        """
        if stop_price_dist <= 0 or symbol_info is None:
            return 0.0
        rf = risk_fraction if risk_fraction is not None else self.cfg["risk_per_trade"]
        rf = min(rf, self.cfg["max_risk_per_trade"])
        risk_amount = equity * rf

        tick_value = getattr(symbol_info, "trade_tick_value", 0.0)
        tick_size = getattr(symbol_info, "trade_tick_size", 0.0)
        if tick_value <= 0 or tick_size <= 0:
            log.warning("%s: missing tick value/size; cannot size.", symbol_info.name)
            return 0.0

        loss_per_lot = (stop_price_dist / tick_size) * tick_value
        if loss_per_lot <= 0:
            return 0.0
        raw = risk_amount / loss_per_lot
        return self._round_volume(symbol_info, raw)

    @staticmethod
    def _round_volume(symbol_info, volume: float) -> float:
        step = getattr(symbol_info, "volume_step", 0.01) or 0.01
        vmin = getattr(symbol_info, "volume_min", 0.01)
        vmax = getattr(symbol_info, "volume_max", 100.0)
        rounded = math.floor(volume / step) * step
        # Fix floating point dust.
        rounded = round(rounded, 8)
        if rounded < vmin:
            return 0.0  # too small to size within risk budget
        return min(rounded, vmax)

    # -- portfolio gates ------------------------------------------------------
    def can_open(self, symbol: str, side: str, positions: list,
                 new_risk_fraction: float) -> tuple[bool, str]:
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"

        if len(positions) >= self.cfg["max_open_positions"]:
            return False, "max_open_positions reached"

        # Portfolio heat: approximate each open position's risk as its
        # configured per-trade risk; cap the sum.
        approx_heat = len(positions) * self.cfg["risk_per_trade"] + new_risk_fraction
        if approx_heat > self.cfg["max_portfolio_heat"]:
            return False, "portfolio heat cap"

        # Correlation: limit same-direction positions within a cluster.
        cluster = self._cluster_for(symbol)
        if cluster:
            same_dir = 0
            for p in positions:
                if p.symbol in cluster and _pos_side(p) == side:
                    same_dir += 1
            if same_dir >= self.cfg["max_correlated_same_dir"]:
                return False, f"correlated same-direction limit ({cluster})"
        return True, ""

    def _cluster_for(self, symbol: str) -> list[str] | None:
        for c in self.clusters:
            if symbol in c:
                return c
        return None

    # -- kill-switch ----------------------------------------------------------
    def check_drawdown(self, equity: float, high_water: float,
                       day_start_equity: float) -> tuple[bool, str]:
        """Return (should_flatten, reason). Also sets the halt flag."""
        if high_water > 0:
            dd = (high_water - equity) / high_water
            if dd >= self.cfg["drawdown_kill_pct"]:
                self.halt(f"max drawdown {dd:.1%} >= {self.cfg['drawdown_kill_pct']:.1%}")
                return True, self.state.halt_reason
        if day_start_equity > 0:
            day_dd = (day_start_equity - equity) / day_start_equity
            if day_dd >= self.cfg["daily_loss_kill_pct"]:
                self.halt(f"daily loss {day_dd:.1%} >= {self.cfg['daily_loss_kill_pct']:.1%}")
                return False, self.state.halt_reason  # halt new, don't force-flatten
        return False, ""

    def halt(self, reason: str) -> None:
        if not self.state.halted:
            log.critical("RISK HALT: %s", reason)
        self.state.halted = True
        self.state.halt_reason = reason

    def reset_daily(self) -> None:
        """Clear a daily-loss halt at the start of a new trading day."""
        if self.state.halted and "daily loss" in self.state.halt_reason:
            log.info("New day: clearing daily-loss halt.")
            self.state.halted = False
            self.state.halt_reason = ""


def _pos_side(position) -> str:
    # MT5 ORDER_TYPE_BUY == 0, ORDER_TYPE_SELL == 1.
    return "buy" if position.type == 0 else "sell"
