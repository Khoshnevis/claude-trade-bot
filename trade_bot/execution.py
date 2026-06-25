"""Execution / OMS layer.

Translates sized signals into MT5 orders and tracks ownership via comment tags
(the bot's magic number already isolates it from other EAs/humans):

  TR|<SYMBOL>            -> trend overlay position
  KP|<A><B>|y / |x       -> the two legs of a Kalman pair

Comment tagging lets a restart re-derive which strategy owns which live
position purely from the broker, so local state is never authoritative.
"""
from __future__ import annotations

import logging

log = logging.getLogger("trade_bot.exec")


def trend_tag(symbol: str) -> str:
    return f"TR|{symbol}"


def pair_tag(a: str, b: str, leg: str) -> str:
    return f"KP|{a}{b}|{leg}"


def _tag_of(position) -> str:
    return (position.comment or "").strip()


class Executor:
    def __init__(self, client):
        self.client = client

    # -- trend ---------------------------------------------------------------
    def open_trend(self, symbol: str, side: str, lots: float,
                   sl: float, tp: float) -> bool:
        res = self.client.market_order(
            symbol, side, lots, sl=sl, tp=tp, comment=trend_tag(symbol)
        )
        return _ok(res)

    def close_tagged(self, tag: str) -> bool:
        """Close every owned position carrying exactly this comment tag."""
        ok = True
        for p in self.client.positions(magic_only=True):
            if _tag_of(p) == tag:
                res = self.client.close_position(p)
                ok = ok and _ok(res)
        return ok

    # -- pairs ---------------------------------------------------------------
    def open_pair(self, a: str, b: str, direction: str,
                  lots_y: float, lots_x: float) -> bool:
        """direction 'long' = long y / short x; 'short' = short y / long x."""
        if direction == "long":
            side_y, side_x = "buy", "sell"
        else:
            side_y, side_x = "sell", "buy"
        r1 = self.client.market_order(a, side_y, lots_y, comment=pair_tag(a, b, "y"))
        if not _ok(r1):
            log.error("Pair %s/%s: y-leg failed; aborting (no x-leg sent).", a, b)
            return False
        r2 = self.client.market_order(b, side_x, lots_x, comment=pair_tag(a, b, "x"))
        if not _ok(r2):
            # Roll back the y leg to stay market-neutral.
            log.error("Pair %s/%s: x-leg failed; unwinding y-leg.", a, b)
            self.close_tagged(pair_tag(a, b, "y"))
            return False
        return True

    def close_pair(self, a: str, b: str) -> bool:
        ok_y = self.close_tagged(pair_tag(a, b, "y"))
        ok_x = self.close_tagged(pair_tag(a, b, "x"))
        return ok_y and ok_x

    # -- introspection -------------------------------------------------------
    def owned_tags(self) -> dict[str, list]:
        out: dict[str, list] = {}
        for p in self.client.positions(magic_only=True):
            out.setdefault(_tag_of(p), []).append(p)
        return out

    def has_trend(self, symbol: str) -> bool:
        return bool(self.owned_tags().get(trend_tag(symbol)))

    def trend_position(self, symbol: str):
        ps = self.owned_tags().get(trend_tag(symbol))
        return ps[0] if ps else None

    def has_pair(self, a: str, b: str) -> bool:
        tags = self.owned_tags()
        return bool(tags.get(pair_tag(a, b, "y")) or tags.get(pair_tag(a, b, "x")))


def _ok(res) -> bool:
    return res is not None and getattr(res, "retcode", None) == 10009  # TRADE_RETCODE_DONE
