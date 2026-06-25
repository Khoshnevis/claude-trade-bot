"""Thin, resilient wrapper around the MetaTrader5 IPC bridge.

The terminal is assumed to be already logged in. ``initialize()`` is therefore
called WITHOUT credentials -- it attaches to the running, authenticated
terminal. Reconnection is handled with exponential backoff.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

try:  # The package only installs/imports on Windows with a terminal present.
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover - allows import on dev machines
    mt5 = None  # type: ignore

log = logging.getLogger("trade_bot.mt5")

# MT5 timeframe name -> constant. Resolved lazily so the module imports even
# when MetaTrader5 is unavailable (e.g. on a Linux dev box).
_TF_NAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]


def timeframe_const(name: str) -> int:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is not available")
    const = getattr(mt5, f"TIMEFRAME_{name.upper()}", None)
    if const is None:
        raise ValueError(f"Unknown timeframe: {name}")
    return const


class MT5Client:
    """Connection manager + typed helpers over the raw MetaTrader5 calls."""

    def __init__(self, terminal_path: str | None = None, magic: int = 0,
                 deviation_points: int = 20):
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 package not importable. Install it on the Windows "
                "host running the (already logged-in) MT5 terminal."
            )
        self.terminal_path = terminal_path
        self.magic = magic
        self.deviation = deviation_points
        self._connected = False

    # -- connection -----------------------------------------------------------
    def connect(self) -> None:
        ok = mt5.initialize(self.terminal_path) if self.terminal_path else mt5.initialize()
        if not ok:
            raise ConnectionError(f"mt5.initialize failed: {mt5.last_error()}")
        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            raise ConnectionError(
                "Connected to terminal but no account is logged in. "
                "Log into the MT5 terminal first."
            )
        self._connected = True
        term = mt5.terminal_info()
        log.info(
            "Connected: account=%s server=%s balance=%.2f %s | terminal=%s build=%s",
            info.login, info.server, info.balance, info.currency,
            getattr(term, "name", "?"), getattr(term, "build", "?"),
        )
        if term is not None and not term.trade_allowed:
            log.warning("Terminal reports trade_allowed=False ('Algo Trading' button off).")

    def shutdown(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False
            log.info("MT5 connection shut down.")

    def ensure_connected(self, max_retries: int = 4) -> None:
        """Reconnect with exponential backoff if the link dropped."""
        if mt5.account_info() is not None:
            return
        log.warning("MT5 link lost; attempting reconnect.")
        self._connected = False
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                mt5.shutdown()
            except Exception:
                pass
            try:
                self.connect()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Reconnect attempt %d/%d failed: %s", attempt, max_retries, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise ConnectionError("Could not re-establish MT5 connection")

    # -- account & symbols ----------------------------------------------------
    def account(self):
        return mt5.account_info()

    def equity(self) -> float:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    def ensure_symbols(self, symbols: list[str]) -> None:
        """Make symbols visible in Market Watch so quotes/orders work."""
        for sym in symbols:
            info = mt5.symbol_info(sym)
            if info is None:
                log.error("Symbol %s not found on this broker.", sym)
                continue
            if not info.visible and not mt5.symbol_select(sym, True):
                log.error("Failed to select symbol %s into Market Watch.", sym)

    def symbol_info(self, symbol: str):
        return mt5.symbol_info(symbol)

    def tick(self, symbol: str):
        return mt5.symbol_info_tick(symbol)

    def server_time(self):
        """Best-effort broker server time from the latest EURUSD-ish tick."""
        t = mt5.symbol_info_tick  # noqa: F841 (placeholder for clarity)
        return None

    # -- market data ----------------------------------------------------------
    def rates(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates for {symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        return df

    # -- positions ------------------------------------------------------------
    def positions(self, magic_only: bool = True) -> list[Any]:
        pos = mt5.positions_get()
        if pos is None:
            return []
        if magic_only:
            return [p for p in pos if p.magic == self.magic]
        return list(pos)

    # -- orders ---------------------------------------------------------------
    def _filling_mode(self, symbol: str) -> int:
        """Pick a supported fill policy for this symbol."""
        info = mt5.symbol_info(symbol)
        mode = getattr(info, "filling_mode", 0) if info else 0
        # filling_mode is a bitmask of supported policies.
        if mode & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        if mode & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def market_order(self, symbol: str, side: str, volume: float,
                     sl: float | None = None, tp: float | None = None,
                     comment: str = "") -> Any:
        """Send a market order. side is 'buy' or 'sell'. Returns the result."""
        self.ensure_symbols([symbol])
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick for {symbol}")
        is_buy = side == "buy"
        price = tick.ask if is_buy else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)
        return self._send_with_retry(request, symbol)

    def close_position(self, position) -> Any:
        symbol = position.symbol
        tick = mt5.symbol_info_tick(symbol)
        is_buy = position.type == mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(position.volume),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        return self._send_with_retry(request, symbol)

    def _send_with_retry(self, request: dict, symbol: str, retries: int = 3) -> Any:
        last = None
        for attempt in range(1, retries + 1):
            result = mt5.order_send(request)
            last = result
            if result is None:
                log.error("order_send returned None: %s", mt5.last_error())
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(
                    "Order OK %s vol=%.2f @%.5f ticket=%s",
                    symbol, request["volume"], result.price, result.order,
                )
                return result
            elif result.retcode in (mt5.TRADE_RETCODE_REQUOTE,
                                    mt5.TRADE_RETCODE_PRICE_OFF,
                                    mt5.TRADE_RETCODE_PRICE_CHANGED):
                # Refresh price and retry.
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None and request["action"] == mt5.TRADE_ACTION_DEAL:
                    is_buy = request["type"] == mt5.ORDER_TYPE_BUY
                    request["price"] = tick.ask if is_buy else tick.bid
                log.warning("Requote/price change (%s), retry %d/%d",
                            result.retcode, attempt, retries)
            else:
                log.error("Order failed %s retcode=%s comment=%s",
                          symbol, result.retcode, result.comment)
                break
            time.sleep(0.5 * attempt)
        return last
