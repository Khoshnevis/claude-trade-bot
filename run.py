#!/usr/bin/env python3
"""Entry point for the MT5 trading bot.

Usage:
    python run.py [path/to/config.yaml]

Assumes the MetaTrader5 terminal is running and ALREADY LOGGED IN. The bot
attaches over the local IPC bridge without credentials.
"""
from __future__ import annotations

import sys

from trade_bot.config import load_config
from trade_bot.engine import TradingEngine
from trade_bot.logging_setup import setup_logging


def main() -> int:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)
    log = setup_logging(cfg["logging"]["level"], cfg["logging"]["file"])
    log.info("Booting trade bot with config: %s", cfg_path)

    engine = TradingEngine(cfg)
    try:
        engine.start()
    except KeyboardInterrupt:
        log.info("Shutdown requested.")
    except Exception as exc:  # noqa: BLE001
        log.exception("Fatal error: %s", exc)
        return 1
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
