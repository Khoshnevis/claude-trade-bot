"""Durable state so a restart resumes cleanly.

Stores the high-water mark, daily anchor equity, and an audit trail of actions.
Open positions themselves are the source of truth on the broker; we reconcile
against them rather than trusting local state.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date

log = logging.getLogger("trade_bot.state")


class StateStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                ts TEXT DEFAULT CURRENT_TIMESTAMP,
                kind TEXT,
                payload TEXT
            );
            """
        )
        self.conn.commit()

    # -- key/value ------------------------------------------------------------
    def get(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    # -- high-water / daily anchor -------------------------------------------
    def high_water(self, equity: float) -> float:
        hw = self.get("high_water", 0.0)
        if equity > hw:
            hw = equity
            self.set("high_water", hw)
        return hw

    def day_anchor(self, equity: float) -> tuple[float, bool]:
        """Return (day_start_equity, is_new_day)."""
        today = date.today().isoformat()
        stored_day = self.get("anchor_day")
        if stored_day != today:
            self.set("anchor_day", today)
            self.set("anchor_equity", equity)
            return equity, True
        return self.get("anchor_equity", equity), False

    # -- audit ----------------------------------------------------------------
    def log_event(self, kind: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO events(kind,payload) VALUES(?,?)",
            (kind, json.dumps(payload, default=str)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
