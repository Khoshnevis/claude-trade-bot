"""Configuration loading and lightweight validation."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "mt5": {"terminal_path": None, "magic": 770088, "deviation_points": 20},
    "engine": {
        "timeframe": "H1",
        "poll_seconds": 15,
        "history_bars": 1500,
        "reconcile_seconds": 300,
        "blackouts": [],
    },
    "risk": {
        "risk_per_trade": 0.005,
        "max_risk_per_trade": 0.01,
        "max_portfolio_heat": 0.02,
        "max_open_positions": 6,
        "max_correlated_same_dir": 1,
        "atr_period": 14,
        "atr_stop_mult": 2.0,
        "atr_target_mult": 3.0,
        "drawdown_kill_pct": 0.12,
        "daily_loss_kill_pct": 0.03,
        "min_volume_guard": True,
    },
    "correlation_clusters": [],
    "pairs": {"enabled": False, "pairs": []},
    "trend": {"enabled": False, "symbols": []},
    "persistence": {"db_path": "state.sqlite"},
    "logging": {"level": "INFO", "file": "trade_bot.log"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def mt5(self) -> dict:
        return self.raw["mt5"]

    @property
    def engine(self) -> dict:
        return self.raw["engine"]

    @property
    def risk(self) -> dict:
        return self.raw["risk"]

    @property
    def pairs(self) -> dict:
        return self.raw["pairs"]

    @property
    def trend(self) -> dict:
        return self.raw["trend"]

    def all_symbols(self) -> list[str]:
        """Every symbol referenced anywhere in the config."""
        symbols: set[str] = set()
        if self.pairs.get("enabled"):
            for p in self.pairs.get("pairs", []):
                symbols.add(p["a"])
                symbols.add(p["b"])
        if self.trend.get("enabled"):
            symbols.update(self.trend.get("symbols", []))
        return sorted(symbols)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    user_cfg: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
    merged = _deep_merge(DEFAULTS, user_cfg)
    _validate(merged)
    return Config(raw=merged)


def _validate(cfg: dict) -> None:
    r = cfg["risk"]
    if not (0 < r["risk_per_trade"] <= r["max_risk_per_trade"]):
        raise ValueError("risk_per_trade must be >0 and <= max_risk_per_trade")
    if not (0 < r["drawdown_kill_pct"] < 1):
        raise ValueError("drawdown_kill_pct must be between 0 and 1")
    if cfg["engine"]["poll_seconds"] <= 0:
        raise ValueError("poll_seconds must be positive")
